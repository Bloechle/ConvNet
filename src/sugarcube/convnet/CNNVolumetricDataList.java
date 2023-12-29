package sugarcube.convnet;

import java.util.LinkedList;

public class CNNVolumetricDataList extends LinkedList<CNNVolumetricData>
{

  public CNNVolumetricDataList()
  {

  }

  public CNNVolumetricDataList dataList(CNNVolumetricData... dataList)
  {
    if (dataList != null)
      for (CNNVolumetricData data : dataList)
        if (data != null)
          this.add(data);
    return this;
  }

  public int volume()
  {
    int size = 0;
    for (CNNVolumetricData data : this)
      size += data.volume;
    return size;
  }

  public void training(boolean on)
  {
    for (CNNVolumetricData cube : this)
      cube.training(on);
  }

}
