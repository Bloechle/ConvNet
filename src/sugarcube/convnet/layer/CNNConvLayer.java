package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNConvLayer extends CNNLocalLayer
{
  public CNNVolumetricData[] filters;
  public CNNVolumetricData biases;
  public double l1DecayMul = 0.0;
  public double l2DecayMul = 1.0;

  public CNNConvLayer(CNNVolumetricData in, int pad, int win, int stride, int features)
  {
    super(CONV, in, pad, win, stride, features);
    this.filters = new CNNVolumetricData[features];
    for (int i = 0; i < features; i++)
      filters[i] = new CNNVolumetricData(win, win, in.depth, Double.NaN);
    this.biases = new CNNVolumetricData(1, 1, features, 0.0).bias();
  }

  @Override
  public void forward(boolean isTraining)
  {
    for (int d = 0; d < out.depth; d++)
    {
      CNNVolumetricData f = filters[d];
      //dummy debug
//      f.toIdentity();
//      Sys.out("f=\n"+f.cubeString());      
      int x = -pad;
      int y = -pad;
      for (int oy = 0; oy < out.height; y += stride, oy++)
      {
        x = -pad;
        for (int ox = 0; ox < out.width; x += stride, ox++)
        {  
          double act = 0.0;
          for (int fy = 0; fy < f.height; fy++)
          {
            int iy = y + fy;
            for (int fx = 0; fx < f.width; fx++)
            {

              int ix = x + fx;
              if (iy >= 0 && ix >= 0 && iy < in.height && ix < in.width)
              {
//                Sys.out(""+ix+","+iy+"-"+fx+","+fy+" ");
                for (int fd = 0; fd < f.depth; fd++)
                  act += f.v[((f.width * fy) + fx) * f.depth + fd] * in.v[((in.width * iy) + ix) * in.depth + fd];
              }
            }
          }
          act += biases.v[d];
          out.setValue(ox, oy, d, act);
        }        
      }
//      Sys.outln("\now="+out.width+", oh="+out.height+", fw="+f.width+", fh="+f.height+", iw="+in.width+", ih="+in.height+", stride="+stride+", pad="+pad);
    }

//    Log.debug(this,  ".forward - out=\n"+out.cubeString());
    super.forward(isTraining);
  }

  @Override
  public void backward()
  {        
    super.backward();
    in.resetGrad();
    for (int d = 0; d < out.depth; d++)
    {
      CNNVolumetricData f = filters[d];
      int x = -pad;
      int y = -pad;
      for (int oy = 0; oy < out.height; y += stride, oy++)
      {
        x = -pad;
        for (int ox = 0; ox < out.width; x += stride, ox++)
        {
          double grad = out.grad(ox, oy, d);
          for (int fy = 0; fy < f.height; fy++)
          {
            int iy = y + fy;
            for (int fx = 0; fx < f.width; fx++)
            {
              int ix = x + fx;
              if (iy >= 0 && ix >= 0 && iy < in.height && ix < in.width)
                for (int fd = 0; fd < f.depth; fd++)
                {
                  int inIndex = ((in.width * iy) + ix) * in.depth + fd;
                  int fIndex = ((f.width * fy) + fx) * f.depth + fd;
                  f.dv[fIndex] += in.v[inIndex] * grad;
                  in.dv[inIndex] += f.v[fIndex] * grad;                  
                }
            }
          }
          this.biases.dv[d] += grad;
        }
      }
//      Sys.outln("f.dv="+f.dArrayString());
    }
  }

  @Override
  public CNNVolumetricData[] filters()
  {
    return filters;
  }

  @Override
  public CNNVolumetricData biases()
  {
    return biases;
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
