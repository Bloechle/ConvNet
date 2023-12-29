package sugarcube.convnet;


import sugarcube.convnet.layer.*;

import java.util.Arrays;
import java.util.Iterator;

public class CNN implements Iterable<CNNLayer>
{
  public static final String IDENTITY = "identity";
  public static final String RELU = "relu";
  public static final String SIGMOID = "sigmoid";
  public static final String TANH = "tanh";
  public static final String DUDA = "duda";

  public static boolean DEBUG = true;
  public CNNLayer[] layers = new CNNLayer[0];

  public CNN()
  {

  }

  public CNN(int width, int height, int depth)
  {
    addInput(width, height, depth);
  }

  public void reset()
  {

  }
  public CNNLayer input()
  {
    return layers[0];
  }

  public CNNLayer output()
  {
    return layers[layers.length - 1];
  }

  public CNNLossLayer outputLoss()
  {
    CNNLayer output = output();
    return output instanceof CNNLossLayer ? (CNNLossLayer) output : null;
  }

  public CNNVolumetricData in()
  {
    return input().in;
  }

  public CNNVolumetricData out()
  {
    return output().out;
  }

  public CNN setInput(double... in)
  {
    for (int i = 0; i < in.length; i++)
      in().v[i] = in[i];
    return this;
  }

  public CNN setGroundtruthIndex(int index)
  {
    outputLoss().groundtruthIndex = index;
    return this;
  }

  public CNNTrainer trainer()
  {
    return new CNNTrainer(this);
  }

  public CNNLayer addLayer(CNNLayer layer)
  {
    return addLayer(layer, null);
  }

  public CNNLayer addLayer(CNNLayer layer, String fct)
  {
    if (fct != null)
      layer.fct = fct;
    CNNLayer[] newLayers = new CNNLayer[layers.length + 1];
    System.arraycopy(layers, 0, newLayers, 0, layers.length);
    newLayers[layers.length] = layer;
    this.layers = newLayers;
    return layer;
  }

  public static double[] toDoubles(float... data)
  {
    if (data == null)
      return null;
    double[] out = new double[data.length];
    for (int y = 0; y < data.length; y++)
      out[y] = data[y];
    return out;
  }

  public void forward(float[] values)
  {
    forward(toDoubles(values));
  }

  public void forward(double[] values)
  {
    this.setInput(values);
    this.forward();
  }

  public void forward()
  {
    forward(false);
  }

  public void forward(boolean isTraining)
  {
    for (int i = 0; i < layers.length; i++)
      layers[i].forward(isTraining);
  }

  public double backward()
  {
    for (int i = layers.length - 1; i >= 0; i--)
      this.layers[i].backward();
    return output().loss();
  }

  public CNNLossLayer setGroundTruth(int index)
  {
    CNNLossLayer output = outputLoss();
    if (output != null)
      output.groundtruthIndex = index;
    return output;
  }

  // public double costLoss(double loss)
  // {
  // this.forward(false);
  // CNNLossLayer last = lastLoss();
  // last.loss = loss;
  // last.backward();
  // return last.loss;
  // }

  public CNNInputLayer addInput(int width, int height, int depth)
  {
    return (CNNInputLayer) addLayer(new CNNInputLayer(width, height, depth));
  }

  public CNNConvLayer addConv(int pad, int win, int stride, int features, String function)
  {
    return addConvolution(pad, win, stride, features, function);
  }

  public CNNConvLayer addConvolution(int pad, int win, int stride, int features, String function)
  {
    CNNConvLayer layer = (CNNConvLayer) addLayer(new CNNConvLayer(out(), pad, win, stride, features), function);    
    if (layer.isFct(CNN.RELU))
      layer.biases.reset(0.1);
    return layer;
  }

  public CNNFullLayer addFull(int neurons, String fct)
  {
    CNNFullLayer layer = (CNNFullLayer) addLayer(new CNNFullLayer(out(), neurons), fct);    
    if (layer.isFct(CNN.RELU))
      layer.biases.reset(0.1);
    return layer;
  }

  public CNNMaxoutLayer addMaxOut(int groupSize)
  {
    return (CNNMaxoutLayer) addLayer(new CNNMaxoutLayer(out(), groupSize));
  }

  public CNNPoolLayer addPool(int pad, int win, int stride)
  {
    return (CNNPoolLayer) addLayer(new CNNPoolLayer(out(), pad, win, stride));
  }

  public CNNDropoutLayer addDrop(double prob)
  {
    return (CNNDropoutLayer) addLayer(new CNNDropoutLayer(out(), prob));
  }

  public CNNSoftMaxLayer addSoftMax(int nbClasses)
  {
    this.addFull(nbClasses, null);
    return (CNNSoftMaxLayer) addLayer(new CNNSoftMaxLayer(out()));
  }

  public CNNSVMLayer addSVM(int nbClasses)
  {
    this.addFull(nbClasses, null);
    return (CNNSVMLayer) addLayer(new CNNSVMLayer(out()));
  }

  public CNNRegressionLayer addRegression(int nbNeurons)
  {
    this.addFull(nbNeurons, null);
    return (CNNRegressionLayer) addLayer(new CNNRegressionLayer(out()));
  }

  @Override
  public Iterator<CNNLayer> iterator()
  {
    return Arrays.asList(layers).iterator();
  }
  
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
