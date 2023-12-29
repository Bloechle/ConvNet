package sugarcube.convnet;

public class CNNDemo
{
  public static void main(String... args)
  {
    CNN cnn = new CNN(24, 24, 1);
    cnn.addConvolution(2, 5, 1, 8, CNN.RELU);
    cnn.addPool(0, 2, 2);
    cnn.addConvolution(2, 5, 1, 16, CNN.RELU);
    cnn.addPool(0, 3, 3);
    cnn.addSoftMax(10);
  }
}
