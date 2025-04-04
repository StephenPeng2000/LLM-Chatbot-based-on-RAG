from zhipuai import ZhipuAI


def get_ans(prompt):
    # 需要修改为自己的智谱的api_key
    client = ZhipuAI(api_key="f12dda7bfde0acc02387d3e9227b9d7e.O2cQ7W5GOZaF2fbT")
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        top_p=0.3,
        temperature=0.45,
        max_tokens=1024,
        stream=True,
    )

    ans = ""
    for trunk in response:
        ans += trunk.choices[0].delta.content
    ans = ans.replace("\n\n", "\n")
    return ans


if __name__ == "__main__":
    prompt = "什么是大模型？"
    get_ans(prompt)
