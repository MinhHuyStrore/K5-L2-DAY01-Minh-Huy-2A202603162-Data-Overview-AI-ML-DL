# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026  
**Runtime Colab:** GPU  
**Python / PyTorch / Ultralytics:** Ultralytics `8.4.145`  
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`  
**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:**

> Báo cáo chỉ sử dụng các file JSON/PNG và thông tin được cung cấp. File `classification_predictions.json` không có trong các tệp tải lên; thông tin hạng 1 của `traffic` được lấy theo record người dùng đã cung cấp trước đó. Output so sánh ngưỡng `0.20 / 0.35 / 0.60` cũng chưa được tải lên, vì vậy không tự suy đoán số prediction ở ngưỡng `0.20`.

## 1. Phân loại ảnh

**Nguồn:** `visuals/classification_top5.png` và record Classification `traffic` đã cung cấp.

1. Record hạng 1:
   - `class_id` = 468
   - `class_name` = cab
   - `rank` = 1
   - `score` = 0.510915
   - `taxonomy_name` = ImageNet-1K
2. Record nói về **cả bức ảnh**, không phải một vật thể riêng. Classification nhận toàn bộ ảnh làm đầu vào và đưa ra các lớp dự đoán cho ảnh. Ảnh traffic có nhiều phương tiện, nhưng biểu đồ top-5 không chỉ ra bounding box hay vị trí của từng phương tiện.
3. Danh sách lớp không phải do model tự nghĩ ra. Các lớp thuộc taxonomy/bộ nhãn được dùng khi huấn luyện và được liên kết với checkpoint. Trong ảnh top-5 này có các tên lớp như `cab`, `minibus`, `police_van`, `recreational_vehicle`, `streetcar`.
4. Cần ghi đủ ba trường vì:
   - `class_id`: mã số định danh lớp, thuận tiện cho máy tính xử lý.
   - `class_name`: tên lớp để con người đọc.
   - `taxonomy_name`: hệ thống phân loại mà mã và tên lớp thuộc về.
   
   Ba trường giúp đối chiếu đúng lớp, tránh nhầm ID giữa các taxonomy.
5. Guideline cần quy định tiêu chí chọn **một lớp đại diện cho toàn ảnh**: chủ thể chính là gì, ưu tiên đối tượng nào khi có nhiều đối tượng, cách xử lý khi nhiều chủ đề ngang nhau hoặc không thể xác định. Trường hợp không rõ cần có nhãn/luồng `ambiguous` hoặc chuyển Reviewer/Lab Coach theo quy định.
6. Không lấy `score` làm ground truth vì score chỉ là mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đúng thực tế. Ground truth phải được người gán nhãn xác định theo nội dung ảnh và guideline, sau đó có thể được Reviewer/QC kiểm tra.

## 2. Phát hiện vật thể

**Nguồn:** `detection_predictions.json`, `visuals/detection_predictions.png`.  
**Sample:** `kitchen` — ảnh kích thước `640 × 427` pixel.  
**Ngưỡng ghi trong JSON:** `0.35`.

1. Một record của `kitchen` (chọn record `person`):
   - `class_name` = `person`
   - `score` = `0.912625`
   - `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
   - `bbox_width` = `113.58`
   - `bbox_height` = `279.68`
2. Với định dạng `xyxy`, tọa độ đầu tiên và thứ hai là góc trên-trái `(x1, y1)`, tọa độ thứ ba và thứ tư là góc dưới-phải `(x2, y2)`. Vì vậy:
   - Góc trên-trái: `(385.33, 69.24)`.
   - Góc dưới-phải: `(498.92, 348.92)`.
   - Box rộng khoảng `113.58` pixel và cao khoảng `279.68` pixel.
   
   Box bao quanh người đứng ở khu vực bên phải ảnh.
3. Số prediction:
   - Ngưỡng `0.35`: có **11 prediction** cho sample `kitchen` trong JSON.
   - Ngưỡng `0.60`: có **6 prediction** có score từ `0.60` trở lên trong danh sách JSON ngưỡng `0.35`.
   - Ngưỡng `0.20`: **chưa xác định được** vì JSON hiện có đã được tạo với ngưỡng `0.35`; các prediction có score dưới `0.35` không được lưu trong file này. Cần output so sánh từ notebook để điền chính xác.
4. Khi hạ ngưỡng, model thường giữ lại nhiều box hơn, bao gồm cả dự đoán yếu hoặc trùng/sai. Reviewer cần kiểm tra box thừa, box trùng, lớp sai, vật thể bị bỏ sót và độ chính xác của vị trí/kích thước; sau đó quyết định giữ, sửa hoặc loại theo guideline.
5. Quy tắc box đề xuất: **Vẽ một bounding box sát nhất có thể quanh toàn bộ phần vật thể nhìn thấy; không cố đoán phần bị che khuất, không lấy phần nền không thuộc vật thể vào box, và nếu vật thể chạm/cắt mép ảnh thì box dừng tại biên ảnh.**
6. Trên ảnh `kitchen`, có một người/phần cơ thể xuất hiện bị cắt ở mép trái, gần góc dưới ảnh. Nếu guideline đã quy định cách gán box cho vật thể bị cắt mép thì làm theo quy tắc đó; nếu chưa quy định hoặc không rõ phần nào thuộc về người, cần escalation lên Lab Coach/Reviewer thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng

**Nguồn:** `segmentation_predictions.json` và ảnh minh họa detection/segmentation được cung cấp.  
**Sample:** `kitchen`.

1. Một record (chọn instance người có score cao nhất):
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.899318`
   - `polygon_point_count` = `348`
   - 5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. Box chỉ cho biết vùng hình chữ nhật bao quanh vật thể. Polygon/mask mô tả đường biên theo hình dạng thực tế của vật thể, nên thể hiện được phần lồi lõm và hình dạng không chữ nhật; nhờ đó phân biệt vùng vật thể với nền chính xác hơn.
3. `instance_id` dùng để phân biệt từng cá thể/vật thể được phát hiện trong một ảnh. Nó không phải `class_id` vì nhiều instance có thể cùng một lớp, ví dụ nhiều chiếc xe đều có class `car`. Nó cũng không phải tracking ID vì mã này không mặc nhiên theo dõi cùng một vật thể qua nhiều frame/video.
4. Quy tắc vẽ mask đề xuất: **Đi theo đường biên của phần vật thể nhìn thấy, bám sát mép vật thể, không lấy nền hoặc vật thể bên cạnh vào mask; không tự vẽ phần bị che khuất nếu guideline không yêu cầu nội suy.**
5. Trên ảnh `kitchen`, các chiếc bát và đồ vật trên bàn nằm gần/chạm nhau, một số vùng biên khó tách rõ khỏi mặt bàn hoặc vật bên cạnh. Nếu guideline đã định nghĩa cách xử lý ranh giới thì áp dụng thống nhất; nếu không thể xác định chính xác biên giữa hai vật, cần escalation để tránh mỗi annotator tự vẽ một kiểu.

## 4. Vòng đời và kiểm tra chất lượng

Chuỗi production: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh, gắn với taxonomy | Ảnh `traffic` có nhiều loại phương tiện; top-5 có nhiều lớp gần nghĩa như `cab`, `minibus`, `police_van` | Chọn một lớp đại diện theo tiêu chí trong guideline; không dùng score làm nhãn chuẩn | Kiểm tra lớp có đúng nội dung/chủ đề chính và đúng taxonomy không |
| Phát hiện vật thể | Mỗi vật thể: class + bounding box `xyxy` | Ảnh `kitchen` có nhiều vật thể nhỏ và gần nhau; có người bị cắt ở mép trái; một số box có thể bao gồm vùng khó phân biệt | Gán đúng lớp và box cho từng vật thể nhìn thấy; áp dụng quy tắc cắt mép/che khuất | Kiểm tra thiếu/thừa box, box trùng, sai lớp, box lệch hoặc quá rộng |
| Instance segmentation | Mỗi instance: class + mask/polygon riêng + instance ID | Biên bát, mặt bàn và vật dụng gần nhau có thể khó tách; vùng bị che khuất làm biên không rõ | Vẽ mask theo biên phần nhìn thấy; giữ instance riêng biệt | Kiểm tra mask có lấn nền/vật khác, thiếu vùng, sai biên hoặc gộp nhầm instance không |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu đúng phạm vi bài lab, không đưa thông tin cá nhân/nhạy cảm vào báo cáo hoặc output và không chia sẻ dữ liệu ra ngoài nơi được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, dừng xử lý và báo cho Lab Coach hoặc người phụ trách dữ liệu để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [ ] `report/REPORT.md` đã điền và kiểm tra.
- [ ] `day1_lab_outputs/classification_predictions.json` có trong repo.
- [ ] `day1_lab_outputs/detection_predictions.json` có trong repo.
- [ ] `day1_lab_outputs/segmentation_predictions.json` có trong repo.
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md` có trong repo.
- [ ] `day1_lab_outputs/visuals/classification_top5.png` có trong repo.
- [ ] `day1_lab_outputs/visuals/detection_predictions.png` có trong repo.
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png` có trong repo.
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác trong báo cáo/output.
- [ ] Đã nộp link repository trên LMS.

## Ghi chú về dữ liệu còn thiếu

- Chưa có file `classification_predictions.json` trong các tệp đính kèm; record `traffic` ở mục 1 dùng thông tin đã được cung cấp trong hội thoại.
- Chưa có output notebook so sánh số prediction ở ngưỡng `0.20 / 0.35 / 0.60`. Mục 2 chỉ xác nhận được `11` prediction ở ngưỡng `0.35` và đếm được `6` record có score từ `0.60` trở lên trong JSON đó. Không thể suy ra số prediction ở ngưỡng `0.20` từ JSON đã lọc ở `0.35`.
- Chưa có thông tin runtime CPU/GPU, phiên bản Python/PyTorch và việc thay đổi notebook nguồn; cần bổ sung sau khi kiểm tra notebook Colab.
- `IMAGE_ATTRIBUTION.md` ghi nguồn ảnh COCO 2017 validation và giấy phép CC BY 2.0 cho các ảnh `traffic`, `kitchen`, `dining`.
