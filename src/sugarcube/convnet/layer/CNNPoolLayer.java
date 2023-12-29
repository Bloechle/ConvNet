package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNPoolLayer extends CNNLocalLayer
{
  // max index neuron map
  public int[] maxX;
  public int[] maxY;

  public CNNPoolLayer(CNNVolumetricData in, int pad, int win, int stride)
  {
    super(POOL, in, pad, win, stride, in.depth);
    this.maxX = new int[out.volume];
    this.maxY = new int[out.volume];
  }

  @Override
  public void forward(boolean isTraining)
  {
    int n = 0;
    for (int d = 0; d < out.depth; d++)
    {
      int x = -pad;
      int y = -pad;
      for (int ox = 0; ox < out.width; x += stride, ox++)
      {
        y = -pad;
        for (int oy = 0; oy < out.height; y += stride, oy++)
        {
          double maxAct = Double.NEGATIVE_INFINITY;
          int maxX = -1;
          int maxY = -1;
          for (int fx = 0; fx < win; fx++)
            for (int fy = 0; fy < win; fy++)
            {
              int iy = y + fy;
              int ix = x + fx;
              if (iy >= 0 && iy < in.height && ix >= 0 && ix < in.width)
              {
                double act = in.value(ix, iy, d);
                if (act > maxAct)
                {
                  maxAct = act;
                  maxX = ix;
                  maxY = iy;
                }
              }
            }

          this.maxX[n] = maxX;
          this.maxY[n] = maxY;
          n++;
          out.setValue(ox, oy, d, maxAct);
        }
      }
    }
    // Log.debug(this, ".forward - \n"+out.cubeString());
  }

  @Override
  public void backward()
  {
    in.resetGrad();
    int n = 0;
    for (int d = 0; d < out.depth; d++)
      for (int ox = 0; ox < out.width; ox++)
        for (int oy = 0; oy < out.height; oy++)
        {
          double grad = out.grad(ox, oy, d);
          in.addGrad(maxX[n], maxY[n], d, grad);
          n++;
        }
  }

}
