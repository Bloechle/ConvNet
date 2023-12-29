package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNRegressionLayer extends CNNLossLayer
{
  public double[] regress;


  public CNNRegressionLayer(CNNVolumetricData in)
  {
    super(REGRESSION, in);
    this.regress = new double[out.v.length];
  }   

  @Override
  public void forward(boolean isTraining)
  {    
    for (int i = 0; i < out.depth; i++)
    {
      regress[i] = out.v[i];
      out.v[i] = in.v[i];
    }
  }

  @Override
  public void backward()
  {
    loss = 0.0;
    for (int i = 0; i < out.depth; i++)
    {
      double dv =  out.v[i]-regress[i];
      in.dv[i] = dv;
//      out.dv[i] = dv;
      loss += 0.5 * dv * dv;
    }
  }
  

}
