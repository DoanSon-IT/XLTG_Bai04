Đoàn Văn Sơn - 20211512 - DCCNTT12.10.6 <br>
#Khi ta để k =5 <br>
SVM - Time: 12.28s, Accuracy: 0.86, Precision: 0.87, Recall: 0.85 <br>
KNN - Time: 0.02s, Accuracy: 0.62, Precision: 0.62, Recall: 0.62 <br>
Decision Tree - Time: 40.56s, Accuracy: 0.79, Precision: 0.79, Recall: 0.78 <br>
#Khi ta để K=3 <br>
SVM - Time: 9.68s, Accuracy: 0.87, Precision: 0.87, Recall: 0.86 <br>
KNN - Time: 0.00s, Accuracy: 0.68, Precision: 0.68, Recall: 0.68 <br>
Decision Tree - Time: 35.85s, Accuracy: 0.82, Precision: 0.81, Recall: 0.82 <br>

- Có 2 folder chứa ảnh Lion và Horse gồm 100 ảnh. 2 folder Lion_augmented và Horse_augmented là ta sử dụng file "them_anh.py" để sinh ra, nhằm đa dạng dữ liệu chuẩn bị cho việc huấn luyện <br>
- File train.py dùng để huấn luyện 3 mô hình KNN, SVM, Tree...<br>
- util.py chứa hàm resize ảnh và lấy ảnh từ 4 folder trên để file train.py gọi lấy dữ liệu <br>
- predict.py dùng để đọc ảnh, dự đoán ảnh <br>

