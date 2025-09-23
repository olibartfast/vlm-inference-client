Sandbox for experimenting with VLM (Vision Language Models)

* Currently developing:
#### OpenAI API-compliant multimodal inference client
  * [C++ code](OpenAI-completion-client/cpp/Readme.md)
  * [Python code](OpenAI-completion-client/python/Readme.md)


# References and other docs
## Some general benchmark sites
* https://mmmu-benchmark.github.io
* https://lmarena.ai
* https://rank.opencompass.org.cn/home (or https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
* https://visioncheckup.com/
* https://playground.roboflow.com/leaderboard

## VideoQA Benchmark
* [MVBench](https://paperswithcode.com/dataset/mvbench)
* [VideoMME](https://video-mme.github.io/home_page.html)
* [TVBench](https://github.com/daniel-cores/tvbench)
* [Action-Atlas](https://mrsalehi.github.io/action-atlas/)
* [sharegpt4video](https://sharegpt4video.github.io/)
* [VideoEval-Pro](https://github.com/TIGER-AI-Lab/VideoEval-Pro)
* [OpenCV - Top VLM Evaluation Metrics for Optimal Performance Analysis](https://learnopencv.com/vlm-evaluation-metrics/)


## Online courses
* https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/
* https://opencv.org/university/vision-language-model-bootcamp


## Vision Multimodal API Services (tentative list)
* https://docs.anthropic.com/en/docs/build-with-claude/vision
* https://platform.openai.com/docs/guides/vision
* https://cookbook.openai.com/topic/multimodal
* https://cloud.google.com/vision?hl=en
* https://azure.microsoft.com/en-us/products/ai-services/ai-vision
* https://aws.amazon.com/rekognition/
* https://platform.stability.ai/docs/api-reference
* https://api.together.ai
* https://cloud.cerebras.ai

## Finetuning 
* Roboflow Maestro:
  * https://github.com/roboflow/maestro/tree/develop/cookbooks
* Unsloth AI:
  * https://docs.unsloth.ai/basics/vision-fine-tuning
  * https://learnopencv.com/unsloth-guide-efficient-llm-fine-tuning/
* Llama factory:
  * https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md
* Huggingface:
  * https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
  * https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
  * https://github.com/merveenoyan/smol-vision
  * [https://huggingface.co/blog/smolvlm2](https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb)
 
## Multimodal RAG
  * https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms
  * https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/ 

## Inference
  ### TensorRT-LLM inference client python/c++
  * TODO check documentation:
     * https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/multimodal/README.md
     * https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/cpp/executor/README.md
  ### Openvino
  * Visual language assistant using: 
     * [DeepSeek-VL2](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/deepseek-vl2/deepseek-vl2.ipynb)
     * [Gemma3](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma3/gemma3.ipynb)
     * [Qwen2.5-VL](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen2.5-vl/qwen2.5-vl.ipynb)

## Use/rent a gpu cluster in cloud
* [brev.nvidia.com](https://docs.nvidia.com/brev/latest/index.html)
* [runpod.io](https://docs.runpod.io/overview)
