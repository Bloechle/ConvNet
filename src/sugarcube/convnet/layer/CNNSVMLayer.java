package sugarcube.convnet.layer;

import sugarcube.convnet.CNN;
import sugarcube.convnet.CNNVolumetricData;

import java.util.Arrays;

public class CNNSVMLayer extends CNNLossLayer
{


  public CNNSVMLayer(CNNVolumetricData in)
  {
    super(SVM, in);
  }

  @Override
  public void forward(boolean isTraining)
  {
    for (int i = 0; i < out.depth; i++)
      out.v[i] = in.v[i];
    if (CNN.DEBUG)
      System.out.println("CNNSVMLayer.forward - out=" + Arrays.toString(out.v));
  }

  @Override
  public void backward()
  {
    // we're using structured loss here, which means that the score
    // of the ground truth should be higher than the score of any other
    // class, by a margin
    loss = 0.0;
    double gt = in.v[groundtruthIndex];
    double margin = 1.0;
    for (int i = 0; i < out.depth; i++)    
      if (i != groundtruthIndex)
      {
        double delta = in.v[i] - gt + margin;
        if (delta > 0)
        {
          // violating dimension, apply loss
          in.dv[i] += 1;
          in.dv[groundtruthIndex] -= 1;
          loss += delta;
        }
      }
    
  }
}
