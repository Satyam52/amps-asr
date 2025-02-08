# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path

import torch
from fairseq2.models.nllb.tokenizer import NllbTokenizer

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
import torch.nn.init as init

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")
import numpy as np
import random
def set_seed(seed:2343):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    op = f"{10*'*'} Seed set to {seed} {10*'*'}"
    logger.info(op)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--s2t_loss_ratio",
        type=int,
    )
    parser.add_argument(
        "--t2t_loss_ratio",
        type=int,
    )
    parser.add_argument(
        "--threshold_loss",
        type=float,
    )
    return parser


def main() -> None:
    seed = 42
    set_seed(seed)
    
    args = init_parser().parse_args()
    dist_utils.init_distributed([logger, trainer.logger])
    device = torch.device("cuda")
    text_tokenizer: NllbTokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer: UnitTokenizer = load_unity_unit_tokenizer(args.model_name)
    finetune_params = trainer.FinetuneParams(
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=device,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        s2t_loss_ratio=args.s2t_loss_ratio,
        t2t_loss_ratio=args.t2t_loss_ratio,
        threshold_loss=args.threshold_loss
    )
    logger.info(f"Finetune params: {finetune_params}")
    model: UnitYModel = load_unity_model(
        args.model_name, device=finetune_params.device, dtype=torch.float16
    )
    logger.info(f"Model {model}")
    
    ############################## Manually intialize adapter_ffn ##############################
    for name, param in model.named_parameters():
        # if "text" not in name:
            if "language_adaptor" in name and "norm" in name:
                if "weight" in name:  # Layer normalization scale
                    print("Initializing {} with mean=0 and std=1".format(name))
                    init.ones_(param.data)
                elif "bias" in name:  # Layer normalization bias
                    print("Initializing {} with mean=0".format(name))
                    init.zeros_(param.data)
                    
            elif "language_adaptor" in name and "bias" in name:
                print("Initializing {}".format(name))
                init.zeros_(param.data)
                        
            elif "language_adaptor" in name and "bias" not in name and "norm" not in name:
                    print("Initializing {}".format(name))
                    init.xavier_uniform_(param.data)   
                    init.xavier_normal_(param.data)
    ############################## Manually intialize adapter_ffn ##############################
    
    
    
    assert model.target_vocab_info == text_tokenizer.vocab_info
    assert model.t2u_model is not None
    assert model.t2u_model.target_vocab_info == unit_tokenizer.vocab_info
    t_mode = "SPEECH_TO_TEXT" if args.mode == trainer.FinetuneMode.SPEECH_TO_TEXT else "SPEECH_TO_PARA"
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
        ),
        dataset_manifest_path=args.train_dataset,
        mode=t_mode
    )
    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
        ),
        dataset_manifest_path=args.eval_dataset,
        mode=t_mode
    )
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
    )
    finetune.run()


if __name__ == "__main__":
    main()
