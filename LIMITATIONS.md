# Limitations

- Latency является proxy-измерением на текущей вычислительной машине.
- Raspberry Pi и OpenMV аппаратно не тестировались.
- TFLite Micro/OpenMV firmware compatibility требует отдельной проверки.
- Датасет содержит однотипные изображения литых изделий; перенос на другое производство требует валидации.
- SOTA comparison условное из-за различий split, preprocessing и числа эпох.
- Average hash для near-duplicates является эвристикой.
