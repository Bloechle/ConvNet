package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNDropoutLayer extends CNNLayer
{
  public double prob;
  public boolean[] dropped;

  public CNNDropoutLayer(CNNVolumetricData in, double prob)
  {
    super(DROPOUT, in);
    this.out = new CNNVolumetricData(in.width, in.height, in.depth, 0.0);
    this.prob = 0.5;
    this.dropped = new boolean[out.volume];
  }

  @Override
  public void forward(boolean isTraining)
  {
    if (isTraining)
    {
      for (int i = 0; i < out.volume; i++)
        if (dropped[i] = Math.random() < prob)
          out.v[i] = 0;
    } else
      for (int i = 0; i < out.volume; i++)
        out.v[i] *= prob;
  }

  @Override
  public void backward()
  {
    for (int i = 0; i < in.volume; i++)
      if (!(dropped[i]))
        in.dv[i] = out.dv[i];
  }

}
