package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNFullLayer extends CNNLayer
{
  public CNNVolumetricData[] weights;
  public CNNVolumetricData biases;
  public double l1DecayMul = 0.0;
  public double l2DecayMul = 1.0;

  public CNNFullLayer(CNNVolumetricData in, int neurons)
  {
    super(FC, in);
    this.out = new CNNVolumetricData(1, 1, neurons, 0.0);
    this.weights = new CNNVolumetricData[neurons];
    for (int i = 0; i < neurons; i++)
      weights[i] = new CNNVolumetricData(1, 1, in.volume, Double.NaN);
    this.biases = new CNNVolumetricData(1, 1, neurons, 0.0).bias();
  }

  @Override
  public void forward(boolean isTraining)
  {
    for (int d = 0; d < out.depth; d++)
    {
      double act = 0.0;
      double[] wd = weights[d].v;
      for (int i = 0; i < in.volume; i++)
        act += in.v[i] * wd[i];
      act += biases.v[d];
      out.v[d] = act;
    }
    super.forward(isTraining);
//    if (CNN.DEBUG)
//      Log.debug(this, ".forward - out=" + Arrays.toString(out.v));
  }

  @Override
  public void backward()
  {
    super.backward();
    in.resetGrad();
    for (int d = 0; d < out.depth; d++)
    {
      CNNVolumetricData wd = weights[d];
      double grad = out.dv[d];
      for (int i = 0; i < in.volume; i++)
      {
        in.dv[i] += wd.v[i] * grad;
        wd.dv[i] += in.v[i] * grad;
      }
      biases.dv[d] += grad;
    }
    
//    if (CNN.DEBUG)
//      for (int i = 0; i < weights.length; i++)
//        Log.debug(this, ".backward - weights=" + Arrays.toString(weights[i].v)+", biases="+Arrays.toString(biases.v));
  }

  @Override
  public CNNVolumetricData[] filters()
  {
    return weights;
  }

  @Override
  public CNNVolumetricData biases()
  {
    return biases;
  }

  public CNNFullLayer biases(double v)
  {
    this.biases.reset(v);
    return this;
  }

  @Override
  public double l1DecayMultiply()
  {
    return l1DecayMul;
  }

  @Override
  public double l2DecayMultiply()
  {
    return l2DecayMul;
  }
}