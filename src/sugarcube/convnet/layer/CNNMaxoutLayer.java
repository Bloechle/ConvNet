package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNMaxoutLayer extends CNNLayer
{
  public int groupSize;
  public int[] maxIndexes;

  public CNNMaxoutLayer(CNNVolumetricData in, int groupSize)
  {
    super(MAXOUT, in);
    this.groupSize = groupSize;
    this.out = new CNNVolumetricData(in.width, in.height, in.depth / groupSize, 0.0);
    this.maxIndexes = new int[out.volume];
  }

  @Override
  public void forward(boolean isTraining)
  {
    // 1D arrays optimization
    if (out.width == 1 && out.height == 1)
    {
      for (int d = 0; d < out.depth; d++)
      {
        int z = d * groupSize; // base index offset
        double max = in.v[z];
        int maxIndex = 0;
        for (int i = 1; i < groupSize; i++)
        {
          double v = in.v[z + i];
          if (v > max)
          {
            max = v;
            maxIndex = i;
          }
        }
        out.v[d] = max;
        maxIndexes[d] = z + maxIndex;
      }
    } else
    {
      int n = 0; // counter for switches
      for (int x = 0; x < in.width; x++)
      {
        for (int y = 0; y < in.height; y++)
        {
          for (int d = 0; d < out.depth; d++)
          {
            int z = d * groupSize;
            double max = in.value(x, y, z);
            int maxIndex = 0;
            for (int i = 1; i < groupSize; i++)
            {
              double v = in.value(x, y, z + i);
              if (v > max)
              {
                max = v;
                maxIndex = i;
              }
            }
            out.setValue(x, y, d, max);
            maxIndexes[n] = z + maxIndex;
            n++;
          }
        }
      }
    }
  }

  @Override
  public void backward()
  {
    if (out.width == 1 && out.height == 1)
    {
      for (int d = 0; d < out.depth; d++)
        in.dv[maxIndexes[d]] = out.dv[d];
    } else
    {
      int n = 0;
      for (int x = 0; x < out.width; x++)
        for (int y = 0; y < out.height; y++)
          for (int d = 0; d < out.depth; d++)
            in.setGrad(x, y, maxIndexes[n++], out.grad(x, y, d));
    }
  }
}