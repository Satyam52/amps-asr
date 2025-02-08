<p align="center">
  <a href="https://github.com/csalt-research">
    <img src="https://avatars.githubusercontent.com/u/43694569?s=200&v=4" alt="CSALT @ IITB" width="150" height="150">
  </a>
  <h3 align="center">AMPS: ASR with Multimodal Paraphrase Supervision</h3>
  <p align="center"> Accepted to NAACL 2025
    <br/>
    <br/>
  </p>
</p>
  

## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [About The Repository](#about-the-repository)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install the dependencies:](#2-install-the-dependencies)
  - [3. Troubleshooting](#3-troubleshooting)
- [Data prepration](#data-prepration)
- [Running experiments](#running-experiments)
- [Inference](#inference)
  - [Steps to create `custom_model.yaml`:](#steps-to-create-custom_modelyaml)
  - [Using the new model for inference:](#using-the-new-model-for-inference)
- [Authors](#authors)
- [Citation](#citation)
- [License](#license)
<!-- * [Citation](#citation) -->

## About The Repository

This repository hosts the code pertaining to our paper [**<samp>AMPS: ASR with Multimodal Paraphrase Supervision</samp>**](https://arxiv.org/abs/2411.18368) accepted to ***NAACL 2025***.

The main contribution of our paper is :mag_right:  a new technique `AMPS` that augments a multilingual multimodal ASR system with paraphrase-based supervision for improved conversational ASR.


## Getting Started

### 1. Clone the Repository  

```bash
git clone https://github.com/csalt-research/amps-asr.git
cd amps-asr
```

### 2. Install the dependencies:

`fairseq2` Dependencies

```bash
cd fairseq2
pip install --editable .
```
`seamless` dependencies:
```bash
cd seamless
pip install --editable .
```
### 3. Troubleshooting
If you encounter any issues while installing dependencies, refer to the [ Installation Guide](https://github.com/facebookresearch/fairseq2/blob/main/INSTALL_FROM_SOURCE.md).

You are all set! 🎉

&nbsp;




---


## Data prepration
Seamless needs dataset in a `json` format. The dataset should be in the following structure:

```json
{"source": {"id": "<ID>", "text": "<T2T-pipeline-input-text>", "lang": "<T2T-pipeline-input-language>", "audio_local_path": "<path-to-audio-file>", "sample_rate": <audio-sample-rate>, "waveform": null, "units": null}, "target": {"id": "<ID>", "text": "<ASR-pipeline-target-text>", "lang": "<ASR+T2T-pipeline-target-language>", "audio_local_path": null, "sample_rate": null, "waveform": null, "units": null, "paraphrase": "<T2T-pipeline-target-paraphrase>"}}
{...}
{...}
{...}
{...}
.
.
.
```
We have provided a sample dataset in the `sample_data` folder



## Running experiments

Our codebase has a simple, easily customizable script `run.sh`, simply execute: 

```bash
./run.sh s2t_loss_ratio t2t_loss_ratio loss_threshold
```
**Note:** The **threshold** is a tunable parameter that can help improve performance. By default, it is set to `-1`, meaning no thresholding is applied.   

For example to run only ASR finetuning without any thresholding, you can execute:

```bash
./run.sh 1 0 -1
```
To run AMPS with **3.2** threshold, you can execute:
```bash
./run.sh 1 1 3.2
```

## Inference

After fine-tuning, the model will be saved in the directory `$EXPERIMENT_DIR`.  
We need to create a new `.yaml` card (let's say `custom_model.yaml`) for the newly fine-tuned model in:  


### Steps to create custom_model.yaml:

1. Copy the content of `BASE_MODEL.yaml` to `custom_model.yaml`.
2. Update the following fields:
   - **Model name**: Change it to `custom_model`.
   - **Checkpoint path**: Set it to `$EXPERIMENT_DIR/$EXPERIMENT_NAME.pt`.

### Using the new model for inference:

Specify the new model in the `model_name` field when using the translator:

```python
# Initialize a Translator object with a new model.
translator = Translator("custom_model", "vocoder_36langs", torch.device("cuda:0"), torch.float16)

# Predict
text_output, _ = translator.predict(
    input=<path_to_input_audio>,
    task_str="ASR",
    tgt_lang=<tgt_lang>,
    text_generation_opts=text_generation_opts,
    unit_generation_opts=None
)
```

For more details on inference, visit [here](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md) 


## Authors

* **Abhishek Gupta** - *MTech, CSE, IIT Bombay* - [Abhishek Gupta](https://www.linkedin.com/in/iam-abhishek/)
* **Amruta Parulekar** - *DD, EE, IIT Bombay* - [Amruta Parulekar]()
* **Sameep Chattopadhyay** - *DD, EE, IIT Bombay* - [Sameep Chattopadhyay]()
* **Preethi Jyothi** - *Associate Professor, CSE, IIT Bombay* - [Preethi Jyothi](https://www.cse.iitb.ac.in/~pjyothi/)

 
## Citation

If you use this code for your research, please consider citing our work.


## License

Distributed under the MIT License. See [LICENSE](https://github.com/csalt-research/accented-codebooks-asr/blob/main/LICENSE.md) for more information.
