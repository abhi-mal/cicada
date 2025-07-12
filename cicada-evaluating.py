import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # because gpu is not needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from pathlib import Path
from tqdm import tqdm
from tensorflow import data
from tensorflow.keras.models import load_model

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image
from drawing import Draw
from generator import RegionETGenerator
import keras
from hgq.layers import QConv2D, QDense, QBatchNormDense
from hgq.config import LayerConfigScope, QuantizerConfigScope

def quantize(arr: npt.NDArray, precision: tuple = (16, 8)) -> npt.NDArray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc

def loss(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    mse_loss = np.mean((y_true - y_pred) ** 2, axis=(1, 2, 3))
    return mse_loss


def main(args):

    config = yaml.safe_load(open(args.config))

    draw = Draw(output_dir=args.output, interactive=args.interactive)

    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    gen_train = gen.get_generator(X_train, X_train, 512, True)
    gen_val = gen.get_generator(X_val, X_val, 512)
    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])

    X_train_student = np.concatenate([X_train, outlier_train])
    X_val_student = np.concatenate([X_val, outlier_train])

    if args.huggingface:
        from huggingface_hub import from_pretrained_keras
        teacher = from_pretrained_keras("cicada-project/teacher-v.0.1")
        cicada_v1 = from_pretrained_keras("cicada-project/cicada-v1.1")
        cicada_v2 = from_pretrained_keras("cicada-project/cicada-v2.1")

    else:
        teacher_layer = keras.layers.TFSMLayer("models_rand_1/teacher", call_endpoint='serving_default') # using pretrained teacher
        input_shape = (18, 14, 1) 
        inputs = keras.Input(shape=input_shape)
        outputs = teacher_layer(inputs)
        teacher = keras.Model(inputs, outputs) # needed because we are loading a old model into keras3
        custom_objects_dict = {
                "QuantizerConfigScope": QuantizerConfigScope,
                "LayerConfigScope": LayerConfigScope,
                "QConv2D": QConv2D,
                "QBatchNormDense": QBatchNormDense,
                "QDense": QDense,
        }
        cicada_v1 = load_model(f"{args.input}/hgq2_model/model_checkpoint.keras",custom_objects=custom_objects_dict)
        cicada_v2 = load_model(f"{args.input}/hgq2_model/model_checkpoint.keras",custom_objects=custom_objects_dict)

        log = pd.read_csv("models_rand_1/teacher/training.log")
        draw.plot_loss_history(
                log["loss"], log["val_loss"], f"training-history-teacher"
            )
        log = pd.read_csv(f"{args.input}/hgq2_model/training.log")
        draw.plot_loss_history(
                log["loss"], log["val_loss"], f"training-history-hgq2_model"
            )            
        #for model in [teacher, cicada_v1, cicada_v2]:
        #    log = pd.read_csv(f"{args.input}/{model.name}/training.log")
        #    draw.plot_loss_history(
        #        log["loss"], log["val_loss"], f"training-history-{model.name}"
        #    )

    # Comparison between original and reconstructed inputs
    X_example = X_test[:1]
    y_example = teacher.predict(X_example, verbose=args.verbose)['teacher_outputs'] # needed because we are loading a old model into keras3
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-background",
    )
    X_example = X_signal["VBFHto2C"][:1]
    y_example = teacher.predict(X_example, verbose=args.verbose)['teacher_outputs']
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-signal",
    )

    # Equivariance plot
    X_example = X_test[0]
    draw.make_equivariance_plot(
        X_example,
        f=lambda x: x[:, ::-1],
        g=lambda x: predict_single_image(teacher, x),
        name='equivariance-plot-mirror'
    )

    # Effect of phi-shifts on anomaly score
    phi_losses = []
    X_example = X_test[:1000]
    for i in tqdm(range(19)):
        X_example_shifted = np.roll(X_example, i, axis=1)
        y_example_shifted = teacher.predict(X_example_shifted, batch_size=512, verbose=args.verbose)['teacher_outputs']
        phi_losses.append(loss(X_example_shifted, y_example_shifted))
    phi_losses = np.array(phi_losses)
    for i in range(phi_losses.shape[1]):
        phi_losses[:, i] = (phi_losses[:, i] - phi_losses[0, i]) / phi_losses[0, i]

    draw.plot_phi_shift_variance(
        phi_losses[:, :1],
        name='loss-variance-phi-teacher-example'
    )

    draw.plot_phi_shift_variance(
        phi_losses,
        name='loss-variance-phi-teacher-average'
    )


    # Evaluation
    y_pred_background_teacher = teacher.predict(X_test, batch_size=512, verbose=args.verbose)['teacher_outputs']
    y_loss_background_teacher = loss(X_test, y_pred_background_teacher)
    y_loss_background_cicada_v1 = cicada_v1.predict(
        X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
    )
    y_loss_background_cicada_v2 = cicada_v2.predict(
        X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
    )

    results_teacher, results_cicada_v1, results_cicada_v2 = dict(), dict(), dict()
    results_teacher["2017 open data Zero Bias (Test)"] = quantize(np.log(y_loss_background_teacher) * 32)# transformation so that we can compare teacher and student anomaly score
    results_cicada_v1["2017 open data Zero Bias (Test)"] = y_loss_background_cicada_v1
    results_cicada_v2["2017 open data Zero Bias (Test)"] = y_loss_background_cicada_v2

    y_true, y_pred_teacher, y_pred_cicada_v1, y_pred_cicada_v2 = [], [], [], []
    inputs = []
    for name, data in X_signal.items():
        inputs.append(np.concatenate((data, X_test)))

        y_loss_teacher = loss(
            data, teacher.predict(data, batch_size=512, verbose=args.verbose)['teacher_outputs']
        )
        y_loss_cicada_v1 = cicada_v1.predict(
            data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
        )
        y_loss_cicada_v2 = cicada_v2.predict(
            data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
        )
        results_teacher[name] = quantize(np.log(y_loss_teacher) * 32)
        results_cicada_v1[name] = y_loss_cicada_v1
        results_cicada_v2[name] = y_loss_cicada_v2

        y_true.append(
            np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
        )
        y_pred_teacher.append(
            np.concatenate((y_loss_teacher, y_loss_background_teacher))
        )
        y_pred_cicada_v1.append(
            np.concatenate((y_loss_cicada_v1, y_loss_background_cicada_v1))
        )
        y_pred_cicada_v2.append(
            np.concatenate((y_loss_cicada_v2, y_loss_background_cicada_v2))
        )

    draw.plot_anomaly_score_distribution(
        list(results_teacher.values()),
        [*results_teacher],
        "anomaly-score-teacher",
    )
    draw.plot_anomaly_score_distribution(
        list(results_cicada_v1.values()),
        [*results_cicada_v1],
        "anomaly-score-cicada-v1",
    )
    draw.plot_anomaly_score_distribution(
        list(results_cicada_v2.values()),
        [*results_cicada_v2],
        "anomaly-score-cicada-v2",
    )

    # ROC Curves with Cross-Validation
    draw.plot_roc_curve(y_true, y_pred_teacher, [*X_signal], inputs, "roc-teacher")
    draw.plot_roc_curve(y_true, y_pred_cicada_v1, [*X_signal], inputs, "roc-cicada-v1")
    draw.plot_roc_curve(y_true, y_pred_cicada_v2, [*X_signal], inputs, "roc-cicada-v2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA evaluation scripts""")
    parser.add_argument(
        "--input", "-i",
        action=IsReadableDir,
        type=Path,
        default="models/",
        help="Path to directory w/ trained models",
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="plots/",
        help="Path to directory where plots will be stored",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Use models from huggingface",
        default=False,
    )
    main(parser.parse_args())
