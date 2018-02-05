package com.tensor.models;
import java.util.Date;

import com.tensor.enums.*;
public class ImageRecognitionTO {
	private String model ="";
	private String imageObjectFileName ="";
	private String imageURL = "";
	
	
	public String getModel() {
		return model;
	}
	public void setModel(String model) {
		this.model = model;
	}
	public String getImageObjectFileName() {
		return imageObjectFileName;
	}
	public void setImageObjectFileName(String imageObjectFileName) {
		this.imageObjectFileName = imageObjectFileName;
	}
	public String getImageURL() {
		return imageURL;
	}
	public void setImageURL(String imageURL) {
		this.imageURL = imageURL;
	}
	
	
	
	
}

