# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
>  Temperature càng gần 0, mô hình càng chọn từ có xác suất cao nhất (câu trả lời ổn định, lặp lại). Temperature cao (gần 1.5) khiến mô hình chọn các từ ít khả dĩ hơn (câu trả lời sáng tạo, đa dạng, hoặc dễ sai lệch).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
>  Chatbot hỗ trợ khách hàng cần sự chính xác và nhất quán để đảm bảo thông tin đúng đắn. Mức temperature thấp (thường 0.0 - 0.2) sẽ phù hợp để giảm thiểu rủi ro mô hình "nói nhảm" hoặc đưa ra thông tin không đúng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
>GPT-4o thường đắt hơn đáng kể so với mini (bạn có thể xem bảng giá API của OpenAI). GPT-4o xứng đáng khi cần logic phức tạp, suy luận sâu, còn mini tối ưu cho các tác vụ nhanh, lặp lại, cần chi phí thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> System prompt thiết lập vai trò và văn phong cho model. Với giáo viên, model dùng ngôn ngữ đơn giản, gần gũi; với chuyên gia, model dùng thuật ngữ kỹ thuật chuyên sâu. Điều này chứng minh system prompt điều khiển hành vi và cách tiếp cận vấn đề của model ngay cả khi câu hỏi đầu vào giống hệt nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi so sánh số lượng token của một đoạn văn tiếng Việt với ước lượng "số từ chia 0.75", kết quả thường cho thấy sự chênh lệch đáng kể, với số token thực tế từ count_tokens lớn hơn từ 20% đến 40% so với con số ước lượng. Nguyên nhân chủ yếu nằm ở cách hoạt động của bộ mã hóa (tokenizer). Các bộ tokenizer phổ biến hiện nay được tối ưu hóa chủ yếu trên tập dữ liệu tiếng Anh, nơi các từ thường được đại diện bởi các token dài và hiệu quả. Ngược lại, do tiếng Việt có các dấu thanh và ký tự đặc thù, bộ mã hóa thường phải bẻ nhỏ các từ này thành nhiều token hơn để biểu diễn, dẫn đến việc tổng số lượng token cho cùng một độ dài văn bản cao hơn so với tiếng Anh. Chính đặc điểm này là yếu tố quan trọng cần cân nhắc khi tính toán chi phí sử dụng API cho các ứng dụng tiếng Việt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi cần giảm độ trễ hiển thị (Time To First Token) cho chatbot để người dùng không cảm thấy sốt ruột. Non-streaming phù hợp hơn cho các hệ thống backend xử lý dữ liệu hàng loạt hoặc các tác vụ cần kết quả hoàn chỉnh ngay lập tức để tiếp tục xử lý logic tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp tránh hiện tượng "cơn bão retry" (thundering herd), nơi hàng nghìn client cùng gửi lại yêu cầu cùng lúc khiến server tiếp tục sập ngay khi vừa hồi phục. Delay tăng dần giúp dàn trải thời gian retry, tạo không gian cho server ổn định và giảm tải hiệu quả hơn so với việc chờ cố định.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Prompt: "Bạn là chuyên gia lập trình Python. Hãy trả lời ngắn gọn, trực diện vào vấn đề và luôn sử dụng tiếng Việt."
Lựa chọn: "Ngắn gọn" giúp tiết kiệm token và tránh nhiễu thông tin trên terminal. "Tiếng Việt" đảm bảo tính nhất quán cho trợ lý.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế: "Mất trí nhớ dài hạn" do giới hạn lịch sử hội thoại (context window).
Cải thiện: Sử dụng "Summarization" — khi lịch sử đầy, yêu cầu mô hình tóm tắt nội dung các lượt cũ thành một đoạn văn ngắn và gộp vào system prompt để duy trì ngữ cảnh.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
