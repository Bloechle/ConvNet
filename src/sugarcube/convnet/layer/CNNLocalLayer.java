package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;

public class CNNLocalLayer extends CNNLayer
{
  //local receptive field
  public int pad;
  public int win;
  public int stride;

  public CNNLocalLayer(String type, CNNVolumetricData in, int pad, int win, int stride, int depth)
  {
    super(type, in);
    this.pad = pad;
    this.win = win;
    this.stride = stride;
    this.out = new CNNVolumetricData(resize(in.width), resize(in.height), depth, 0.0);
  }

  public int resize(int size)
  {
    return (int) Math.floor((size + pad * 2.0 - win) / stride + 1.0);
  }

}
