# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`) -> (468, 'cab', 1, 0.510915, 'ImageNet-1K')
  
- Record này mô tả toàn ảnh như thế nào?
-> Khi đối chiếu với file classification_top5.jpg, nhãn "cab" (xe taxi) có điểm số cao nhất nhưng không thể hiện được toàn bộ bối cảnh của bức ảnh. Bức ảnh thực tế là một khung cảnh đường phố phức tạp với nhiều loại phương tiện khác nhau (xe buýt, ô tô con) và người đi bộ. Vì mô hình phân loại (image classification) chỉ đánh giá và xếp hạng nhãn cho toàn bộ bức ảnh thay vì nhận diện từng vật thể cụ thể, nó đã chọn ra đặc trưng của một nhóm vật thể nổi bật để đại diện, dẫn đến việc bỏ sót ngữ cảnh tổng thể của giao thông đô thị.
  
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
-> Danh sách lớp được quyết định bởi những nhà nghiên cứu và kỹ sư tạo ra bộ dữ liệu huấn luyện gốc. Trong trường hợp này, danh sách lớp thuộc về những người xây dựng bộ dữ liệu ImageNet (với taxonomy ImageNet-1K).
  
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
-> Class ID: Định dạng số nguyên giúp tối ưu hóa không gian lưu trữ và tăng tốc độ xử lý tính toán của máy tính.
-> Tên lớp (Class name): Cung cấp thông tin dạng text có ý nghĩa để con người (người gán nhãn, chuyên viên phân tích dữ liệu) có thể đọc hiểu trực quan kết quả.
-> Taxonomy: Đóng vai trò là "hệ quy chiếu". Các bộ dữ liệu khác nhau (như COCO, ImageNet) có bộ tiêu chuẩn khác nhau. Lưu taxonomy giúp đảm bảo ID và tên lớp được diễn giải chính xác, tránh việc ID 468 của taxonomy này bị nhầm với ID 468 của taxonomy khác.
  
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
-> Với bài toán phân loại đơn nhãn (single-label), guideline cần đưa ra quy tắc ưu tiên rõ ràng. Ví dụ: chỉ gán nhãn cho vật thể chiếm diện tích lớn nhất, vật thể nằm ở vị trí trung tâm, hoặc ưu tiên nhãn bối cảnh (scene) hơn là vật thể đơn lẻ. Nếu tính chất dữ liệu thường xuyên có nhiều chủ thể, guideline cần hướng dẫn người gán nhãn chuyển sang phương pháp phân loại đa nhãn (multi-label) hoặc bài toán phát hiện vật thể (object detection).
  
- Vì sao model score không phải ground truth?
-> Model score chỉ là mức độ tự tin (confidence probability) mà thuật toán tính toán ra dựa trên những gì nó học được. Trong khi đó, ground truth là sự thật khách quan (nhãn chuẩn) do con người chủ động gán ghép và xác nhận. Một mô hình có thể dự đoán một lớp với điểm số rất cao (ví dụ score = 0.99) nhưng nhãn đó vẫn có thể hoàn toàn sai so với thực tế (ground truth).

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  -> class_name: person
  -> score: 0.912625
  -> bbox_xyxy: (385.33, 69.24, 498.92, 348.92)
  -> bbox_width: 113.58
  -> bbox_height: 279.68
- Diễn giải vị trí box bằng lời:
  -> Dựa vào tọa độ, bounding box này bắt đầu từ điểm góc trên bên trái ở tọa độ (x=385.33, y=69.24) và kết thúc ở góc dưới bên phải tại (x=498.92, y=348.92). Với chiều rộng 113.58 pixel và chiều cao 279.68 pixel, hộp này tạo thành một hình chữ nhật nằm ở nửa bên phải bức ảnh, bao bọc vừa vặn hình dáng của người đàn ông mặc tạp dề trắng đang quay lưng lại bồn rửa.
  
- So sánh số prediction ở hai threshold:
-> Ở ngưỡng score_threshold = 0.35, mô hình phát hiện được 11 vật thể trong ảnh nhà bếp (bao gồm người, bát, lò nướng, cốc).
-> Nếu ta áp dụng mức ngưỡng cao hơn là score_threshold = 0.70 (lọc bỏ các bản ghi có điểm dưới 0.7), số lượng dự đoán giảm mạnh xuống chỉ còn 3 vật thể (cụ thể là person 0.91, bowl 0.71 và bowl 0.70).
  
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
-> Khi hạ ngưỡng, độ bao phủ (Recall) tăng vì mô hình quét được nhiều vật thể hơn, ít bỏ sót. Tuy nhiên, khối lượng công việc của reviewer sẽ tăng vọt do họ phải tự tay xóa bỏ các bounding box bị nhiễu hoặc sai (False Positives). Khi tăng ngưỡng, độ chính xác (Precision) tăng. Reviewer sẽ nhàn hơn ở khâu lọc nhiễu, nhưng lại phải đối mặt với rủi ro phải tự vẽ thêm (draw from scratch) rất nhiều bounding box cho các vật thể thực tế có tồn tại nhưng bị mô hình bỏ sót do điểm số dưới ngưỡng.
  
- Đề xuất một quy tắc box chặt:
-> Một bounding box được coi là "chặt" khi 4 cạnh của nó (trên, dưới, trái, phải) chạm sát nhất có thể vào các điểm ảnh (pixel) ngoài cùng của vật thể, không để chừa lại quá nhiều không gian trống (background) bên trong hộp. Quy tắc cốt lõi: Bao trọn toàn bộ vật thể nhưng không được cắt lẹm vào vật thể và hạn chế tối đa không gian thừa.
  
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
-> Vật thể bị che khuất (Occlusion): Guideline cần quy định rõ annotator phải vẽ bounding box bao quanh phần nhìn thấy được (visible part), hay phải ước lượng để vẽ bao trọn toàn bộ kích thước thực của vật thể (kể cả phần bị che). Ngoài ra, cần mức chặn tỷ lệ (ví dụ: bị che trên 70% thì không gán nhãn). Vật thể bị cắt mép (Truncation): Guideline cần định nghĩa mức độ lộ diện tối thiểu trong khung hình. Ví dụ, một cái đĩa chỉ lộ 10% ở rìa ảnh thì có cần vẽ box hay không? Escalation: Nếu annotator gặp trường hợp các vật thể chồng chéo lên nhau quá phức tạp hoặc mờ ảo không thể phân định ranh giới (như đống nồi niêu lộn xộn trong ảnh), họ cần có quy trình báo cáo (escalate) cho team lead hoặc chuyên gia QA để chốt phương án xử lý chung, tránh việc mỗi người gán nhãn một kiểu làm hỏng chất lượng dữ liệu huấn luyện.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: segmentation_predictions.json và visuals/segmentation_prediction.png, sample kitchen.

- *Một record:* instance_id="kitchen-001", class_name="person", score=0.899318, polygon_point_count=348; năm điểm đầu của polygon_xy là [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]].
- *Chi tiết polygon so với box:* Box chỉ cho biết hình chữ nhật bao ngoài. Polygon đi theo hình dạng nhìn thấy của người, loại phần nền trong các góc box và cho phép tính vùng pixel của instance chính xác hơn.
- *Vai trò của* instance_id**:** kitchen-001 định danh riêng instance này trong output. Hai object cùng lớp vẫn có hai instance_id và hai polygon khác nhau. Đây không phải class_id và cũng không phải tracking ID xuyên video.
- *Quy tắc biên mask:* Bám theo pixel biên nhìn thấy cuối cùng của object; không lấy bóng đổ, nền hoặc object chạm cạnh. Các lỗ và phần bị che khuất phải xử lý thống nhất theo guideline, không tự nội suy phần không nhìn thấy.
- *Vùng mờ, tiếp xúc hoặc che khuất:* Guideline phải quy định include/exclude cho vùng bán trong suốt, biên mờ, lỗ và các object chạm nhau. Nếu không thể xác định biên hoặc lớp từ ảnh, annotator đánh dấu và chuyển reviewer. Quan sát thực tế cho thấy vùng lá treo ở góc trên trái được model dự đoán là potted plant (kitchen-004, score 0.632236), nên reviewer phải kiểm tra lại class và biên thay vì chép prediction thành ground truth.

# 4. Vòng đời và kiểm tra chất lượng

Chuỗi ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework bắt đầu từ ảnh hợp lệ trong phạm vi. Guideline biến yêu cầu thành quy tắc gán nhãn; annotator tạo ground truth theo quy tắc đó. Ground truth đã QC được dùng để huấn luyện model. Prediction của model sau huấn luyện tiếp tục được so với guideline và bằng chứng trực quan; lỗi, thiếu hoặc mơ hồ được chuyển sang QC và rework, không được coi prediction là sự thật.


| Tác vụ                | Đơn vị/định dạng ground truth                                         | Lỗi hoặc điểm mơ hồ quan sát được                                                                                    | Annotator làm gì?                                                                                  | Reviewer xem gì?                                                                                          |
| --------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một ảnh và một class_id theo taxonomy nếu guideline là single-label | Ảnh traffic có nhiều loại xe nhưng top-1 là cab; chủ thể chính không duy nhất                                    | Áp dụng quy tắc chọn chủ thể chính hoặc đánh dấu ảnh mơ hồ/out-of-scope                            | Kiểm tra class theo toàn ảnh, taxonomy và cách xử lý ảnh nhiều chủ thể; không dùng score làm ground truth |
| Phát hiện vật thể     | Một record class_id + bbox_xyxy cho mỗi object                      | Ở mép trái chỉ thấy một phần người nhưng model vẫn tạo box person; các vật thể nhỏ trên bàn dễ bị nhầm hoặc bỏ sót | Gán mỗi object một box chặt riêng, đánh dấu che khuất/cắt mép và không bỏ object chỉ vì score thấp | Kiểm tra lớp, box thừa nền, box trùng, object bỏ sót và tính nhất quán của cờ che khuất/cắt mép           |
| Instance segmentation | Một record class_id + instance_id + polygon_xy cho mỗi instance     | Lá treo góc trên trái bị dự đoán là potted plant; biên các vật thể chạm nhau trên bàn khó tách                     | Vẽ mask theo phần nhìn thấy, tách từng instance và chuyển vùng không phân biệt được để review      | Kiểm tra sai class, rò mask sang nền/object khác, lỗ, biên tiếp xúc và tính nhất quán giữa các instance   |


## 5. An toàn dữ liệu

- *Một quy tắc bảo vệ dữ liệu:* Chỉ dùng ba ảnh COCO công khai đã được notebook cố định và kiểm tra SHA-256; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng, dữ liệu nội bộ hoặc thông tin nhạy cảm lên Colab/GitHub công khai.
- *Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi:* Tôi sẽ dừng xử lý, không sao chép hoặc công khai thêm, và báo cho Lab Coach hoặc giảng viên phụ trách để xác nhận cách xử lý.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
