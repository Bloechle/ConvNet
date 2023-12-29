package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;
import sugarcube.convnet.CNNUtil;

public class CNNSoftMaxLayer extends CNNLossLayer
{
  public double[] exp;  

  public CNNSoftMaxLayer(CNNVolumetricData in)
  {
    super(SOFTMAX, in);
    this.exp = new double[out.depth];
  }

  @Override
  public void forward(boolean isTraining)
  {    
    double max = CNNUtil.Max(in.v);
    double sum = 0.0;
    for (int i = 0; i < out.depth; i++)
    {
      double e = Math.exp(in.v[i] - max);
      sum += e;
      exp[i] = e;
    }

    // normalize and output to sum to one
    for (int i = 0; i < out.depth; i++)
    {
      exp[i] /= sum;
      out.v[i] = exp[i];
    }
//    if (CNN.DEBUG)
//      Log.debug(this, ".forward - out=" + Arrays.toString(out.v));
  }

  @Override
  public void backward()
  {
    // compute and accumulate gradient wrt weights and bias of this layer
    for (int i = 0; i < out.depth; i++)
      in.dv[i] = -((i == groundtruthIndex ? 1.0 : 0.0) - exp[i]);
    
    // loss is the class negative log likelihood
    loss = -Math.log(exp[groundtruthIndex]);      
  }

}
