# Reflection — Lab 19

**Tên:** Nguyễn Thị Hạ
**MSSV:** 2A202602536
**Cohort:** A20-K4
**Path đã chạy:** lite (WSL Ubuntu, Python 3.12, `BAAI/bge-small-en-v1.5`)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 trung bình: hybrid **78.6%** > BM25 77.8% > vector 73.2%.

- **exact:** BM25 = hybrid = 96.7%, vector 88.7%. Thuật ngữ xuất hiện nguyên
  văn trong doc nên khớp từ khoá là đủ.
- **mixed:** hybrid **100%** > vector 98.5% > BM25 97.0%. RRF cộng thứ hạng
  của hai nguồn, nên doc nào cả hai cùng xếp cao sẽ nổi lên đầu.
- **paraphrase:** cả ba đều yếu (BM25 33.3%, hybrid 32.0%, vector 24.0%).
  Vector **không** thắng như kỳ vọng vì `bge-small-en` là model tiếng Anh,
  không hiểu câu tiếng Việt diễn đạt lại. Lỗi nằm ở việc chọn model, không
  phải ở việc fusion.

Khi nào **không** dùng hybrid:

- **Pure BM25:** tra mã lỗi, SKU, tên hàm, số hiệu điều luật. Cần khớp chính
  xác và cần P99 thấp (BM25 2.8 ms so với hybrid 21 ms).
- **Pure vector:** khi embedding model đa ngữ tốt (bge-m3) mà query chủ yếu là
  ngôn ngữ tự nhiên hoặc đa ngữ. Lúc đó BM25 chỉ thêm nhiễu và tốn thêm độ trễ.

---

## Điều ngạc nhiên nhất khi làm lab này

Ở ngưỡng 0.75 (con số AWS công bố), semantic cache trả **36% câu trả lời sai**
trên corpus này. Phải lên 0.85 thì tỉ lệ sai mới về 0% mà vẫn tiết kiệm 100%.
Ngoài ra, chỉ cần quên filter tenant là GLOBEX đọc được doanh thu của ACME, mà
hệ thống không báo một lỗi nào.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: (làm một mình)
