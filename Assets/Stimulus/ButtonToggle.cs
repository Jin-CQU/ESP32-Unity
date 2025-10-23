using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ButtonToggle : MonoBehaviour
{
    public GameObject Image1;

    public GameObject Image2;

    public GameObject Image3;

    public void ToggleImage()
    {
        if (Image1 != null)
        {
            Image1.SetActive(!Image1.activeSelf);
        }
        if (Image2 != null)
        {
            Image2.SetActive(!Image2.activeSelf);
        }
        if (Image3 != null)
        {
            Image3.SetActive(!Image3.activeSelf);
        }
    }
}
