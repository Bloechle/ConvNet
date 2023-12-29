package sugarcube.convnet;

public interface ICNNSample
{
   int classIndex();
  
   void feed(CNNVolumetricData cube);
}
