import settings as se

print(se.FILE_TRAIN_DATA)
print(se.FILE_COUNTS_DATA)
for i in range(0, 10):
    print(se.FILE_TIME_WINDOW_X(i))
print(se.FILE_NORMALIZATION_DATA)
print(se.FILE_WINDOWED_DATA)
print(se.FILE_MODEL)