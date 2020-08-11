import withcnn
import noncnn
x = input("Try with or without CNN, Enter a for CNN, other number for non-CNN: ")
if x == 'a':
    withcnn.main()
else:
    noncnn.main()

