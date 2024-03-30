import os

from modal import Image, Secret, Stub, enter, exit, gpu, method

MODEL_DIR = "/model"
BASE_MODEL = "jjovalle99/starling-7b-raft-ft"
GPU_CONFIG = gpu.A10G(count=1)


# Download the model
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf", "*.bin"],
    )
    move_cache()


# Image definition
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm==0.3.2",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=60 * 20,
    )
)

stub = Stub("raft-starling7b-ft", image=image)


# Encapulate the model in a class
@stub.cls(gpu=GPU_CONFIG, secrets=[Secret.from_name("huggingface-secret")], allow_concurrent_inputs=10)
class Model:
    @enter()
    def load(self):
        from vllm import LLM

        if GPU_CONFIG.count > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)

        self.template = "<s>GPT4 Correct User: {user}<|end_of_turn|>GPT4 Correct Assistant:"

        self.llm = LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
        )

    @method()
    def generate(self, user_questions):
        import time

        from vllm import SamplingParams

        prompts = [
            self.llm.get_tokenizer().apply_chat_template(
                conversation=[{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            for q in user_questions
        ]
        # prompts = [self.template.format(user=q) for q in user_questions]

        sampling_params = SamplingParams(
            temperature=1e-5,
            top_p=0.9,
            top_k=100,
            max_tokens=500,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {BASE_MODEL} in {duration_s:.1f} seconds, throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )
        return result

    @exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "What are the pilars of the Well-Architected Framework?",
        "What is the AWS Shared Responsibility Model?",
    ]
    model.generate.remote(questions)


