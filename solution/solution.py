"""
K4 — Ngày 1: Khám Phá LLM API (4 tiếng)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: phút 60–100)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI
 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    # TODO: gọi call_openai với model=OPENAI_MINI_MODEL
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    raise NotImplementedError("Implement call_openai_mini")


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_text, gpt4o_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)
    cost = (len(gpt4o_text.split()) / 0.75) / 1000 \
           * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }
    raise NotImplementedError("Implement compare_models")


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: phút 100–140)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency
    raise NotImplementedError("Implement chat_with_system_prompt")


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Nhánh dự phòng khi model lạ hoặc lỗi tải encoding
        return max(1, len(text) // 4)
    raise NotImplementedError("Implement count_tokens")


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(input_te: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        input_cost = input_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    # TODO: đếm token prompt/response, tra bảng giá, trả về dict 5 key

    # Định nghĩa bảng giá (giá ví dụ per 1K tokens)
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
    }

    # Lấy giá, dùng .get() để có giá trị dự phòng nếu model không có trong bảng
    model_price = PRICING.get(model, {"input": 0.00015, "output": 0.0006})

    # Đếm token
    in_tokens = count_tokens(input_te, model)
    out_tokens = count_tokens(response, model)

    # Tính chi phí
    in_cost = (in_tokens / 1000) * model_price["input"]
    out_cost = (out_tokens / 1000) * model_price["output"]

    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": in_cost + out_cost
    }
    raise NotImplementedError("Implement estimate_cost")


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: phút 150–190)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        history = []
        while True:
            user_msg = input("Bạn: ")
            if user_msg.strip().lower() in ("quit", "exit"):
                break
            messages = history + [{"role": "user", "content": user_msg}]
            stream = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True,
            )
            reply = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                reply += delta
            print()
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
            history = history[-6:]
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))

# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: phút 190–230)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    if get_input is None:
        get_input = input
    from openai import OpenAI
 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        messages = ([{"role": "system", "content": persona}]
                    + history + [{"role": "user", "content": user_msg}])
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True,
            )
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]
    return {"num_turns": num_turns, "total_tokens": total_tokens,
            "total_cost": total_cost, "history": history}


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    # TODO (bonus): lặp qua prompts, gọi compare_models, thêm key "prompt"
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({"prompt": prompt, **comparison})
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    # TODO (bonus): dựng chuỗi bảng và trả về
    def shorten(value: Any, limit: int = 40) -> str:
        """Convert a value to a compact, single-line table cell."""
        text = str(value).replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    headers = [
        "Prompt", "GPT-4o Response", "Mini Response",
        "GPT-4o Latency", "Mini Latency",
    ]
    rows = [headers]
    for result in results:
        rows.append([
            shorten(result.get("prompt", "")),
            shorten(result.get("gpt4o_response", "")),
            shorten(result.get("mini_response", "")),
            f'{result.get("gpt4o_latency", 0):.2f}s',
            f'{result.get("mini_latency", 0):.2f}s',
        ])

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator]
    for index, row in enumerate(rows):
        lines.append("| " + " | ".join(
            cell.ljust(widths[column]) for column, cell in enumerate(row)
        ) + " |")
        if index == 0:
            lines.append(separator)
    lines.append(separator)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
