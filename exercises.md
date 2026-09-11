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
> Càng tăng temperature, độ sáng tạo càng tăng, văn phong càng ngẫu nhiên và có thể đôi lúc đi chệch khỏi sự thật, nhưng bù lại mang lại những cách diễn đạt mới lạ.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đặt temperature thấp (khoảng 0.0 đến 0.2) cho chatbot hỗ trợ khách hàng. Vì cần phản hồi chính xác, nhất quán và tuân thủ chặt chẽ các chính sách, tránh cung cấp thông tin sai lệch cho khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Ước tính chi phí
Theo giá hiện tại: GPT-4o output $10/1M token, GPT-4o-mini output $0.60/1M token → GPT-4o đắt hơn khoảng 16-17 lần trên output (và ~17 lần trên input, $2.50 vs $0.15/1M). Với workload này: 10.000 người × 3 lần/ngày × 350 token = 10,5 triệu token output/ngày. GPT-4o ≈ $105/ngày (~$3.150/tháng), GPT-4o-mini ≈ $6.3/ngày (~$190/tháng) — chênh lệch khoảng $2.960/tháng chỉ riêng phần output.
(Lưu ý: sang 2026 OpenAI đã coi GPT-4o là model legacy, không còn niêm yết công khai, khuyến nghị dùng GPT-4.1/GPT-5 series cho tích hợp mới — nhưng logic đánh đổi chi phí giữa bản "full" và bản "mini" vẫn tương tự.)
Khi nào GPT-4o (bản full) xứng đáng: Các tác vụ cần suy luận nhiều bước, tổng hợp thông tin phức tạp, hoặc độ chính xác cao ảnh hưởng trực tiếp đến quyết định người dùng (ví dụ: phân tích y tế, tư vấn pháp lý, code review phức tạp) — nơi một câu trả lời sai lệch gây thiệt hại lớn hơn nhiều so với khoản chênh lệch chi phí.
Khi nào nên dùng mini: Các tác vụ đơn giản, có tính lặp lại cao — phân loại intent, trích xuất thông tin có cấu trúc, trả lời FAQ, tóm tắt ngắn — nơi chất lượng mini gần như tương đương full model nhưng tần suất gọi rất cao (3 lần/user/ngày × 10.000 user là quy mô điển hình cần tối ưu chi phí per-call).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với role giáo viên tiểu học, phản hồi ngắn gọn, dùng từ vựng đơn giản, ví dụ gần gũi. Với role chuyên gia tài chính, phản hồi dài hơn, dùng nhiều thuật ngữ kỹ thuật chuyên sâu. System prompt đóng vai trò định hướng persona, giúp model điều chỉnh phong cách, ngữ điệu và độ phức tạp của câu trả lời sao cho phù hợp với đối tượng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tiếng Việt có nhiều từ ghép và dấu thanh nên bị tách thành nhiều token hơn (đôi khi theo từng âm tiết thay vì cả từ nguyên vẹn). Số token tiktoken đếm thường cao hơn số ước lượng khoảng 30-50% so với tiếng Anh, do bộ mã hoá (BPE) ưu tiên cho tiếng Anh hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang chờ trực tiếp và độ trễ cảm nhận (perceived latency) ảnh hưởng đến trải nghiệm — ví dụ chatbot, trợ lý viết code, hay bất kỳ giao diện hội thoại nào. Với response dài (vài trăm đến vài nghìn token), non-streaming bắt người dùng nhìn màn hình trống 5-10 giây rồi nhận toàn bộ câu trả lời một lúc; streaming cho họ thấy chữ xuất hiện ngay sau ~1 giây đầu, nên cảm giác "nhanh" hơn nhiều dù tổng thời gian xử lý gần như không đổi. Ngược lại, non-streaming phù hợp hơn khi: (1) hệ thống xử lý ở backend không có người chờ trực tiếp (batch job, pipeline tự động), (2) bạn cần parse toàn bộ output thành JSON/structured data trước khi dùng — nhận từng phần rồi mới ráp lại chỉ làm code phức tạp hơn mà không lợi gì, hoặc (3) output rất ngắn (vài từ, một câu trả lời có/không) thì streaming gần như không tạo khác biệt về UX nhưng lại tốn thêm công sức implement.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Với delay cố định, nếu server đang quá tải, tất cả client retry sau đúng 1 giây sẽ đồng loạt gửi lại request cùng một lúc — tạo ra một đợt sóng traffic (thundering herd) khiến server càng quá tải hơn, dẫn đến retry tiếp theo cũng thất bại, và vòng lặp lặp lại vô hạn mà không bao giờ giảm tải. Exponential backoff giải quyết việc này bằng cách tăng dần thời gian chờ sau mỗi lần thất bại (ví dụ: 1s → 2s → 4s → 8s...), nên nếu nhiều client cùng gặp lỗi cùng lúc, các lần retry của họ sẽ dần dần dãn ra và lệch nhau theo thời gian thay vì dồn cục — cho server cơ hội phục hồi. Thực tế thường kết hợp thêm jitter (thêm một khoảng random nhỏ vào mỗi lần delay) để tránh trường hợp nhiều client có cùng lịch retry (do cùng bắt đầu request gần như đồng thời) vẫn vô tình đồng bộ với nhau dù đã dùng backoff.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Chọn persona: "Trợ lý hỗ trợ tra cứu tài liệu học tập" — hội thoại ngắn, trả lời dựa trên ngữ cảnh người dùng cung cấp (ví dụ: đoạn text, câu hỏi ôn tập), không đóng vai chuyên gia tuyệt đối.

Bạn là trợ lý học tập, giúp người dùng ôn tập và tra cứu nhanh kiến thức
từ tài liệu họ cung cấp.

- Trả lời ngắn gọn, tối đa 3-4 câu cho câu hỏi đơn giản; chỉ mở rộng khi
  người dùng yêu cầu "giải thích chi tiết hơn".
- Luôn trả lời bằng ngôn ngữ người dùng dùng để hỏi.
- Nếu câu trả lời dựa trên tài liệu người dùng gửi, trích dẫn phần liên
  quan; nếu dựa trên kiến thức chung, nói rõ "theo hiểu biết chung, không
  phải từ tài liệu bạn gửi".
- Nếu không tìm thấy thông tin liên quan trong tài liệu, nói rõ thay vì
  suy đoán hoặc bịa nội dung.

Giải thích 2 lựa chọn từ ngữ:

"Trả lời ngắn gọn... chỉ mở rộng khi người dùng yêu cầu" — ràng buộc độ dài mặc định để tránh model tự động viết dài dòng (xu hướng tự nhiên của LLM là giải thích thừa). Đặt điều kiện "trừ khi yêu cầu" thay vì cấm tuyệt đối giúp giữ tính linh hoạt — người dùng vẫn có thể chủ động xin chi tiết hơn khi cần ôn sâu.
"Nếu dựa trên kiến thức chung, nói rõ 'theo hiểu biết chung, không phải từ tài liệu bạn gửi'" — đây là lựa chọn từ ngữ quan trọng nhất để giảm ảo giác (hallucination) bị hiểu nhầm là trích từ nguồn. Nếu không tách bạch rõ, người dùng dễ tưởng mọi câu trả lời đều có căn cứ từ tài liệu họ đưa, dẫn đến tin sai thông tin khi ôn thi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: không có bộ nhớ dài hạn giữa các phiên — mỗi lần mở lại, trợ lý không nhớ người dùng đã hỏi gì, tài liệu nào đã gửi trước đó, nên phải lặp lại ngữ cảnh từ đầu mỗi phiên mới.

Đề xuất cải thiện: lưu tóm tắt ngữ cảnh phiên (session summary) vào một file/database bên ngoài, inject lại vào system prompt ở đầu phiên mới.

Cách triển khai ngắn gọn:

Sau mỗi phiên (hoặc mỗi N lượt hội thoại), gọi thêm 1 lần API với prompt "tóm tắt những gì người dùng đang học và câu hỏi họ hay hỏi trong 2-3 câu" → lưu kết quả vào key-value store (key = user_id).
Đầu phiên mới, đọc summary tương ứng, chèn vào system prompt dạng: Ngữ cảnh từ phiên trước: {summary}.
Giới hạn kích thước summary (ví dụ dưới 200 từ) và ghi đè (không cộng dồn vô hạn) để tránh phình context theo thời gian — tương tự cách một memory layer thực tế thường làm: nén dần thay vì lưu toàn bộ lịch sử thô.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
