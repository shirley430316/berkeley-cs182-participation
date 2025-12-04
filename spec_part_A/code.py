# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import markdown
from datetime import datetime

question1 = """
请务必用**中文**思考，并用英文回答以下问题。
### **5. Fermi Estimation for Large-scale Deep Learning Models**
In this question, you will perform **Fermi estimates** for a hypothetical GPT-6 model with **100 trillion parameters** (10¹⁴). You’ll explore scaling laws, memory, inference, cost, and environmental impact using rough, order-of-magnitude calculations. Assume 16-bit parameters (2 bytes each) unless stated otherwise.
#### **(a) Compute and dataset scaling**
The Chinchilla scaling laws relate model size, dataset size, and training compute. For a given compute budget \( C \) in FLOP, the optimal parameter count \( N \) and token count \( D \) scale as \( N = 0.1 C^{0.5} \) and \( D = 1.7 C^{0.5} \). If GPT-6 has \( 10^{14} \) parameters, **what training compute \( C \) is required, and how large must the training dataset \( D \) be?**
#### **(b) Dataset size in human terms**
To contextualize the dataset size from (a), assume each English word corresponds to about 1.4 tokens, each page contains 400 words, and each book has 300 pages. **How many books would the training dataset correspond to?** Compare this number to the size of the Library of Congress (roughly 20 million volumes).
#### **(c) Memory requirements**
Each 16-bit parameter occupies 2 bytes. **How much memory (in GB or TB) is required to store GPT-6’s 100 trillion parameters?** Given that an H200 GPU has about 100 GB of VRAM, **how many such GPUs would be needed just to hold the model in memory?**
#### **(d) Inference latency and throughput**
During inference, model parameters must be loaded from GPU memory. The H200 has a memory bandwidth of 4.8 TB/s. **What is the minimal time in seconds to perform one forward pass through GPT-6?** If the model generates tokens autoregressively (one token per forward pass), **how many tokens could it output in one minute?**
#### **(e) Training cost in FLOPs and dollars**
Training compute is often measured in petaFLOP-days. One petaFLOP-day equals about \( 8.64 \times 10^{19} \) FLOP. GPT-3 required 3640 petaFLOP-days to train. **If trained on H200 GPUs (each delivering 1.98 petaFLOP/s and renting for $1.50/hour), how much would it cost to train GPT-3?** Using your computed \( C \) from part (a), **estimate the cost to train GPT-6 under the same assumptions.**
#### **(f) Inference cost and break-even**
For Transformer models, inference requires about 2 FLOPs per parameter per token. **How many FLOPs are needed to generate 1 million tokens with a 1-trillion-parameter model like GPT-5?** If OpenAI charges $120 per million tokens, **how many tokens must be sold to recoup a $1 billion training cost?** Express this in terms of 1000-word essays, assuming 1.4 tokens per word.
#### **(g) Environmental impact**
Training GPT-3 emitted roughly 552 tonnes of CO₂. The social cost of carbon is around $112 per tonne. **What is the carbon cost of training GPT-3 in USD?** For comparison, producing 1 kg of beef emits about 50 kg of CO₂. A quarter-pound burger contains about 113 g of beef. **How many burgers’ worth of CO₂ does training GPT-3 represent?**
*Use order-of-magnitude arithmetic, scientific notation, and clear reasoning. Show all steps concisely.*
"""
question1 = question1.strip()

def main():
    client = OpenAI(
        api_key='?????????????',
        base_url="https://api.deepseek.com")


    print("🚀 Sending request to DeepSeek API...")

    start_time = datetime.now()
    raw_response = client.chat.completions.with_raw_response.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question1},
        ],
        stream=False
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"⏱️ Time taken for request: {duration:.2f} seconds\n")

    response = raw_response.parse()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cn/deepseek_output_{timestamp}.txt"

    print(f"📝 output file: {output_file}\n")

    reasoning_content = response.choices[0].message.reasoning_content or ""
    final_answer = response.choices[0].message.content or ""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("HTTP request head\n")
        f.write("=" * 60 + "\n")
        for header, value in raw_response.headers.items():
            f.write(f"{header}: {value}\n")
    
        # 写入请求时间戳
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: deepseek-reasoner\n")
        f.write("=" * 60 + "\n\n")
    
        f.write("=" * 60 + "\n")
        f.write("🤔 Reasoning Content\n")
        f.write("=" * 60 + "\n")
        if reasoning_content:
            f.write(reasoning_content)
            f.write("\n")
        else:
            f.write(" No reasoning\n")
    
        f.write("\n" + "=" * 60 + "\n\n")
    
        f.write("=" * 60 + "\n")
        f.write("💡 Final Answer\n")
        f.write("=" * 60 + "\n")
        if final_answer:
            f.write(final_answer)
        else:
            f.write(" No final Answer\n")
    
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("📊 Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total time for request: {duration:.2f} seconds\n")

        reasoning_chars = len(reasoning_content)
        answer_chars = len(final_answer)
        total_chars = reasoning_chars + answer_chars
    
        f.write(f"Character count - Reasoning: {reasoning_chars}, Answer: {answer_chars}, Total: {total_chars}\n")
    
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            f.write("Token usage:\n")
            f.write(f"   Prompt Tokens (输入): {getattr(usage, 'prompt_tokens', 'N/A')}\n")
            f.write(f"   Completion Tokens (输出): {getattr(usage, 'completion_tokens', 'N/A')}\n")
            f.write(f"   Total Tokens (总计): {getattr(usage, 'total_tokens', 'N/A')}\n")

    print(f"✅ Output written to {output_file}")

    print("\n" + "=" * 60)
    print("Preview of Output")
    print("=" * 60)

    if reasoning_content:
        print("\n🤔 Reasoning: (First 200 characters)")
        print(reasoning_content[:200] + ("..." if len(reasoning_content) > 200 else ""))
    else:
        print("\n🤔 Reasoning: NULL")

    if final_answer:
        print("\n💡 Final Answer: (First 200 characters)")
        print(final_answer[:200] + ("..." if len(final_answer) > 200 else ""))
    else:
        print("\n💡 Final Answer: NULL")

    print(f"Character count - Reasoning: {reasoning_chars}, Answer: {answer_chars}, Total: {total_chars}\n")


for i in range(10):
    main()