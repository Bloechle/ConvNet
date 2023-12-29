package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNInputLayer extends CNNLayer
{

  public CNNInputLayer(int width, int height, int depth)
  {
    super(INPUT, new CNNVolumetricData(width, height, depth, 0.0));
    out = in;
  }

  @Override
  public void forward(boolean isTraining)
  {
    super.forward(isTraining);
  }

  @Override
  public void backward()
  {
    super.backward();
  }

}