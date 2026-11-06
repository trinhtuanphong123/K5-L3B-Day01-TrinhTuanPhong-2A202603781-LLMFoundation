# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603781*
- **Họ và tên:*Trịnh Tuấn Phong*
- **SEED** (in ở ô đầu notebook): 2729

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Thế hệ / Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| **1. BoW + LogReg (tự code)** | 0.907 | 0.889 | 0.08 ms | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.936 | 0.947 | 0.20 ms | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.902 | 0.950 | 40.53 ms | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.966 | 1.000 | 29.35 ms | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.954 | 0.974 | 35.97 ms | $0.000 |
| **4a. LLM nhỏ zero-shot** | 0.000 | 0.000 | 1216.68 ms | $0.011 |
| **4b. LLM lớn zero-shot** | 0.000 | 0.000 | 1262.61 ms | $0.105 |
| **4c. LLM nhỏ few-shot** | 0.000 | 0.000 | 1682.26 ms | $0.022 |

*Lưu ý: Các hàng LLM ở thí nghiệm trước đó trả về F1 = 0 do lỗi giới hạn độ dài `max_tokens=3`, khiến JSON/nhãn bị cắt cụt và không parse được. Sau khi sửa lỗi ở phần Thử thách cá nhân, hiệu năng thực tế của LLM đạt tối đa F1 = 1.0.*

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất:

1. **Dự đoán `llm_models` (F1 của LLM):**
   - *Bạn đoán:* Model lớn và nhỏ sẽ có F1 cao và hoạt động tốt.
   - *Thực tế:* F1 ban đầu trả về 0.000.
   - *Lý do lệch:* Cấu hình hệ thống giới hạn nghiêm ngặt `max_tokens=3`, trong khi các mô hình sinh cần nhiều token hơn để tạo phản hồi theo định dạng. Output bị cắt cụt khiến hàm parse không đọc được nhãn.
2. **Dự đoán `llm_fewshot` (Few-shot cải thiện F1):**
   - *Bạn đoán:* Thêm ví dụ mẫu (few-shot) sẽ nâng cao F1 đáng kể.
   - *Thực tế:* F1 vẫn là 0.000 do lỗi cắt cụt token tương tự.
   - *Lý do lệch:* Cùng rào cản kỹ thuật về giới hạn chiều dài output tối đa của API chưa được tinh chỉnh tương thích.
3. **Dự đoán `emb_search` (Tìm kiếm ngữ nghĩa):**
   - *Bạn đoán:* Truy vấn né từ khóa sẽ trả về kết quả gần nhất là `ham`.
   - *Thực tế:* Trả về 6 kết quả gần nhất đều là `spam` với độ tương đồng rất cao (~0.54).
   - *Lý do lệch:* Câu truy vấn mồi nhử (*"complimentary holiday... claim"*) có ngữ nghĩa rất sát với hành vi tặng quà/trúng thưởng của tin nhắn rác. Vì vậy, mô hình embedding nhận diện ý đồ rác dù không trùng từ khóa bề mặt.

## 4. Phân tích 50 case khó

- **Mô hình tụt dốc nhiều nhất:** `1. BoW + LogReg` (F1 giảm mạnh từ **0.907** ở tập thường xuống **0.286** ở tập khó).
- **Nhóm case gây khó khăn nhất:** **Spam giọng tự nhiên (natural_spam)**, khi tất cả các mô hình đều nhận diện sai hoàn toàn (tỷ lệ đúng 0% ở nhóm này).

**Hai case cụ thể khiến mô hình thất bại:**

1. **Case #12 (Spam giọng tự nhiên):** *"hi there, sorry to bother you, my friend gave me your number... check my profile at meetlocal.club"*
   - *Cơ chế sai:* Tin nhắn mở đầu lịch sự, giả vờ làm quen xã giao và không chứa các từ khóa nhạy cảm thường gặp trong bộ từ điển rác (như *free, prize, claim*). Vì vậy, mô hình BoW bị đánh lừa.
2. **Case #29 (Né từ khóa bằng Obfuscation):** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd..."*
   - *Cơ chế sai:* Thay thế chữ cái bằng ký tự số tương đồng (o -> 0, i -> 1) làm phân mảnh từ gốc. Các token như `c0ngr4tulat10ns` hay `g1ft` trở thành từ ngoài từ vựng (out-of-vocabulary) trong từ điển BoW 2000 từ, khiến tín hiệu spam của các từ khóa này bị triệt tiêu.

## 5. Thử thách cá nhân

- **Đề #6:** Sửa `build_classification_prompt` để LLM trả về cấu trúc JSON `{"label": ..., "reason": ...}`, viết hàm parse và chạy thử nghiệm trên 10 mẫu thuộc tập SUB.
- **Cách làm:**
  - Cập nhật prompt hệ thống, yêu cầu chỉ trả về cấu trúc JSON hợp lệ.
  - Tăng giới hạn an toàn `max_tokens` từ `12` lên `128` để tránh cắt cụt chuỗi.
  - Dùng thư viện `json` kết hợp biểu thức chính quy (Regex) để loại bỏ rào cản Markdown (nếu có) và bóc tách chính xác các trường dữ liệu.
- **Kết quả:**
  - **Tỉ lệ parse thành công:** 100% (10/10 mẫu).
  - **Hiệu năng:** F1 = **1.0** và Accuracy = **1.0**.
  - **Chi phí & token:** Chi phí tăng gấp 2–3 lần so với phương án trả về một từ, do output có thêm trường lý do giải thích.
- **Giải thích:** JSON hữu ích khi hệ thống cần tính minh bạch, chẳng hạn giải thích lý do phân loại cho khách hàng hoặc kiểm toán viên. Tuy nhiên, cách này không tối ưu về chi phí và tốc độ cho các luồng xử lý thời gian thực có lưu lượng rất lớn.

## 6. Phương án cho sản phẩm thật

- **Lựa chọn tối ưu:** **3b. Encoder (MiniLM) + Logistic Regression**.
- **Số liệu chứng minh:**
  - **Chất lượng cao:** F1 (test) = **0.966** và F1 (SUB 40) = **1.000**.
  - **Tốc độ đáp ứng SLA:** Độ trễ p95 là **29.35 ms**, nằm trong giới hạn SLA < 50 ms.
  - **Chi phí vận hành tối giản:** Chi phí API là **$0.00** do mô hình được tự host và chạy cục bộ.
- **Sự đánh đổi:** Chấp nhận thời gian huấn luyện ban đầu (khoảng 56.4 giây để biến đổi vector tập nền) và cần tài nguyên phần cứng (RAM/GPU nhẹ) để lưu trữ, tính toán vector embedding 384 chiều của MiniLM.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
