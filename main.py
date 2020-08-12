import withcnn
import noncnn
import nponly

print("Welcome to Memes vs Notes Classifier")
print("Select Your Choice of Training Algorithm:")
print("a. TensorFlow based using CNN")
print("b. TensorFlow based using Fully Connected Layers Only")
print("c. Only NumPy based using Fully Connected Layers Only")
x = input("Enter Your Choice: ")

if x == 'a':
    withcnn.main()
elif x == 'b':
    noncnn.main()
elif x == 'c':
    nponly.main()

