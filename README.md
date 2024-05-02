<p align="center">
    <img src="assets/retroformer-banner.png" alt="Retroformer.com" />
  </a>
</p>

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/trl/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/trl/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
</p>
      
<p align="center">
  <a href="https://retroformer.github.io/"><strong>[🏠Website & Demo]</strong></a> 
  |<a href="https://openreview.net/pdf?id=KOZu91CzbK"><strong>[📄ICLR Spotlight]</strong></a> |
<a href="https://huggingface.co/papers/2308.02151"><strong>[🤗HF Paper]</strong></a> |
<a href="https://huggingface.co/datasets/Salesforce/RLHF-Retroformer"><strong>[📊Datasets]</strong></a> |
<a href="https://huggingface.co/Salesforce/Retroformer"><strong>[🤖Models]</strong></a> |
<a href="mailto:weiran.yao@salesforce.com"><strong>[📧 Contact Us]</strong></a>
</p>

## 👋 Overview <a name="overview"></a>
This paper introduces a principled framework for reinforcing large language agents by learning a retrospective model, which automatically tunes the language agent prompts from environment feedback through policy gradient. Specifically, our proposed agent architecture learns from rewards across multiple environments and tasks, for fine-tuning a pre-trained language model which refines the language agent prompt by summarizing the root cause of prior failed attempts and proposing action plans. Experimental results on various tasks demonstrate that the language agents improve over time and that our approach considerably outperforms baselines that do not properly leverage gradients from the environment.

<p align="center">
  <img src="assets/arch.png" style="width: 80%; height: auto;">
</p>

### ✨ Framework <a name="aci"></a>
Retroformer is comprised of two language model components: an **actor** LLM, denoted, which generates reasoning thoughts and actions, and a **retrospective** LLM, which generates verbal reinforcement cues to assist the actor in self-improvement by refining the actor prompt with reflection responses. 

The actor model is regarded as an frozen LLM, such as GPT, with inaccessible model parameters. In this scenario, the most direct approach to enhancing actor performance in a given environment is by refining the actor LM's prompt. Consequently, the retrospective model, a smaller local language model, refines the actor's prompt by incorporating a concise summary of errors and valuable insights from failed attempts. We therefore aim to optimize the retrospective model using environment reward. The desired behavior of $M_r$ is to improve the actor model $M_a$ in next attempt. Hence, the difference in episode returns between two consecutive trials naturally serves as a reward signal for fine-tuning the retrospective model with reinforcement learning. 

Read our paper for more details.

```
@article{yao2023retroformer,
  title={Retroformer: Retrospective large language agents with policy gradient optimization},
  author={Yao, Weiran and Heinecke, Shelby and Niebles, Juan Carlos and Liu, Zhiwei and Feng, Yihao and Xue, Le and Murthy, Rithesh and Chen, Zeyuan and Zhang, Jianguo and Arpit, Devansh and others},
  journal={arXiv preprint arXiv:2308.02151},
  year={2023}
}
```

## 🚀 Setup <a name="setup"></a>
1. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/), then create the `retroformer` environment with `conda create -n retroformer python=3.10 -y`
3. Activate using `conda activate retroformer`.
4. Run `./setup.sh` to create the `retroformer` docker image.
5. Create a `.env` file at the root of this repository and fill in the following:

```
OPENAI_API_KEY='OpenAI API Key Here if using OpenAI Model (required for inference)'
OPENAI_MODEL='OpenAI MODEL NAME'
```

## 💽 Usage <a name="usage"></a>
There are two steps to the Retroformer pipeline. First Retroformer takes an input GitHub issue and returns a pull request that attempts to fix it. We call that step *inference*. The second step (currently, only available for issues in the SWE-bench benchmark) is to *evaluate* the pull request to verify that it has indeed fixed the issue. 

_NOTE_: At this moment, there are known issues with a small number of repositories that don't install properly for `arm64` / `aarch64` architecture computers. We're working on a fix, but if you'd like to run and evaluate on the entirety of SWE-bench, the easiest way is by using an `x86` machine.

### 👩‍💻 Inference <a name="inference"></a>

**Inference on AgentBoard Benchmark**: Run Retroformer on [SWE-bench Lite](https://www.swebench.com/lite.html) and generate patches.
```
python run.py --model_name gpt4 \
  --per_instance_cost_limit 2.00 \
  --config_file ./config/default.yaml
```

If you'd like to run on a *single* environment from AgentBoard, such as `webshop`, use the `--environment` option as follows:
```
python run.py --model_name gpt4 \
  --environment webshop
```
* See the [`scripts/`](scripts/) folder for other useful scripts and details.
* See the [`configs/`](config/) folder for details about how you can define your own configuration!
* See the [`AgentBoard/agent/`](sweagent/agent/) folder for details about the logic behind configuration based workflows.
* See the [`data/`](trajectories) folder for details about the output of `run.py`.


## 💫 Contributions <a name="contributions"></a>
- If you'd like to ask questions, learn about upcoming features, and participate in future development, join our [Discord community](https://discord.gg/AVEFbBn2rH)!
- If you'd like to contribute to the codebase, we welcome [issues](https://github.com/princeton-nlp/Retroformer/issues) and [pull requests](https://github.com/MetaMind/Retroformer/pulls)!

Contact person: [Weiran Yao](mailto:weiran.yao@salesforce.com). 

## 🪪 License <a name="license"></a>
MIT. Check `LICENSE`.
