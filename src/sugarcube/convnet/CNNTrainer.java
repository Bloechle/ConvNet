package sugarcube.convnet;

import sugarcube.convnet.layer.CNNLayer;

public class CNNTrainer
{
  public static final String SGD = "sgd";
  public static final String ADAM = "adam";
  public static final String ADAGRAD = "adagrad";
  public static final String WINDOWGRAD = "windowgrad";
  public static final String ADADELTA = "adadelta";
  public static final String NESTEROV = "nesterov";

  public String method = SGD;
  public double learningRate = 0.01;
  public double l1Decay = 0.0;
  public double l2Decay = 0.0001;
  public int batchSize = 1;
  public double momentum = 0.9;
  public double ro = 0.95;// adadelta
  public double eps = 1e-8;// adam adadelta
  public double beta1 = 0.9;// adam
  public double beta2 = 0.999;// adam
  public int iteration = 0;
//  public double[][] gsum;// last gradients (used for momementum calculation)
//  public double[][] xsum;// adam adadelta
  public boolean regression = false;
  public CNN cnn;
  public boolean initialized = false;
  public ICNNListener listener = null;
//  public double[] lastInput = null;
  public transient double costLoss = 0.0;
  public transient double l1DecayLoss = 0.0;
  public transient double l2DecayLoss = 0.0;

  public CNNTrainer(CNN cnn)
  {
    this.cnn = cnn;
    this.regression = cnn.output().type.equals(CNNLayer.REGRESSION);
    this.reset();
  }

  public CNNTrainer listener(ICNNListener listener)
  {
    this.listener = listener;
    return this;
  }

  public CNNTrainer method(String method)
  {
    this.method = method;
    return this;
  }

  public void reset()
  {
    this.iteration = 0;
    for (CNNLayer layer : cnn)
      layer.weights().training(true);
  }

  public void dispose()
  {
    for (CNNLayer layer : cnn)
      layer.weights().training(false);
  }

//  public void train(SampleSet trainSet, int cycles, boolean reset)
//  {
//    if (reset)
//      this.reset();
//    for (int i = 0; i < cycles; i++)
//    {
//      for (Sample sample : trainSet)
//      {
//        int out = sample.label().equals("red") ? 0 : 1;
//        train(out, sample.doubles());
//      }
//    }
//  }

  public double rate(Iterable<? extends ICNNSample> samples)
  {
    int counter = 0;
    int success = 0;
    for (ICNNSample smp : samples)
    {
      smp.feed(cnn.in());
      cnn.forward();
      if (cnn.outputLoss().maxIndex() == smp.classIndex())
        success++;
      counter++;
    }
    return counter == 0 ? 0 : success / (double) counter;
  }

  public void train(ICNNSample smp)
  {
    smp.feed(cnn.in());
    cnn.setGroundTruth(smp.classIndex());
    this.train();
  }

  public void train(int groundtruthIndex, double... in)
  {
    cnn.setInput(in);
    cnn.setGroundtruthIndex(groundtruthIndex);
    this.train();
  }
  
  public boolean isBatchIteration()
  {
    return iteration % batchSize == 0;
  }

  public boolean train()
  {
    // Log.debug(this, ".train - in=" + Arrays.toString(net.in().v) + ", out=" +
    // net.outputLoss().groundtruthIndex);

    this.cnn.forward(true);
    this.costLoss = cnn.backward();        
    this.l2DecayLoss = 0.0;
    this.l1DecayLoss = 0.0;
    
    // if (CNN.DEBUG)
    // Log.debug(this, ".train - backward in " + bwdTime);
    // if(regression && y.constructor !== Array)
    // console.log("Warning: a regression net requires an array as training
    // output vector.");

    this.iteration++;    
    if (isBatchIteration())
    {
      // perform an update for all sets of weights
      for (int i = 0; i < cnn.layers.length; i++)
      {
        CNNLayer layer = cnn.layers[i];
        CNNVolumetricDataList weights = layer.weights();
        if (!weights.isEmpty())
          for (CNNVolumetricData vol : weights)
          {
            double[] w = vol.v;
            double[] dw = vol.dv;
            double[] gSum = vol.gSum;
            double[] xSum = vol.xSum;
            double l2DecayLayer = vol.isBias ? 0 : l2Decay * layer.l2DecayMultiply();
            double l1DecayLayer = vol.isBias ? 0 : l1Decay * layer.l1DecayMultiply();

            for (int j = 0; j < w.length; j++)
            {
              // accumulate weight decay loss
              l2DecayLoss += l2DecayLayer * w[j] * w[j] / 2;
              l1DecayLoss += l1DecayLayer * Math.abs(w[j]);
              double l1grad = l1DecayLayer * (w[j] > 0 ? 1 : -1);
              double l2grad = l2DecayLayer * w[j];

              // raw batch gradient
              double grad = (l2grad + l1grad + dw[j]) / batchSize;
              double dx;

              switch (method)
              {
              case ADAM:
                // update biased first moment estimate
                gSum[j] = gSum[j] * beta1 + (1 - beta1) * grad;
                // update biased second moment estimate
                xSum[j] = xSum[j] * beta2 + (1 - beta2) * grad * grad;
                // correct bias first moment estimate
                double biasCorr1 = gSum[j] * (1 - Math.pow(beta1, iteration));
                // correct bias second moment estimate
                double biasCorr2 = xSum[j] * (1 - Math.pow(beta2, iteration));
                dx = -learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + eps);
                w[j] += dx;
                break;
              case ADAGRAD:
                gSum[j] = gSum[j] + grad * grad;
                dx = -learningRate / Math.sqrt(gSum[j] + eps) * grad;
                w[j] += dx;
                break;
              case WINDOWGRAD:
                // this is adagrad but with a moving window weighted average
                // so the gradient is not accumulated over the entire history of
                // the run.
                // it's also referred to as Idea #1 in Zeiler paper on Adadelta.
                // Seems reasonable to me!
                gSum[j] = ro * gSum[j] + (1 - ro) * grad * grad;
                // eps added for better conditioning
                dx = -learningRate / Math.sqrt(gSum[j] + eps) * grad;
                w[j] += dx;
                break;
              case ADADELTA:
                gSum[j] = ro * gSum[j] + (1 - ro) * grad * grad;
                dx = -Math.sqrt((xSum[j] + eps) / (gSum[j] + eps)) * grad;
                // yes, xsum lags behind gsum by 1.
                xSum[j] = ro * xSum[j] + (1 - ro) * dx * dx;
                w[j] += dx;
                // System.out.println("dx="+dx+", w="+w[j]);
                break;
              case NESTEROV:
                dx = gSum[j];
                gSum[j] = gSum[j] * momentum + learningRate * grad;
                dx = momentum * dx - (1.0 + momentum) * gSum[j];
                w[j] += dx;
                break;
              default:
                // assume SGD
                if (this.momentum > 0.0)
                {
                  // momentum update
                  dx = momentum * gSum[j] - learningRate * grad; // step
                  gSum[j] = dx; // back this up for next iteration of momentum
                  w[j] += dx; // apply corrected gradient
                } else
                {
                  // vanilla sgd
                  w[j] += -learningRate * grad;
                }
                break;
              }
              // zero out gradient so that we can begin accumulating anew
              dw[j] = 0.0;
            }
          }
      }

      if (listener != null)
        listener.batchTrained();
      return true;
    }
    // if (CNN.DEBUG)
    // Log.debug(this, ".train - train in " + trainTime);
    return false;
  }
}
