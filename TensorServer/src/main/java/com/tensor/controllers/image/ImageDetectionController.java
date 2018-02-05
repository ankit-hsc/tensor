/**
**********************************************************************************************************
--  FILENAME		: ImageDetectionController.java
--  DESCRIPTION		: Controller Class Image detection
--
--  Copyright		: Copyright (c) 2018.
--  Company			: 
--
--  Revision History
-- --------------------------------------------------------------------------------------------------------
-- |VERSION |      Date                              |      Author              |      Reason for Changes                                         |
-- --------------------------------------------------------------------------------------------------------
-- |  0.1   |   Feb 01, 2018                         |      Ankit Mohanty       |       Initial draft                                                |
-- --------------------------------------------------------------------------------------------------------
--
************************************************************************************************************
**/

package com.tensor.controllers.image;



import java.io.IOException;
import java.util.List;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;


import org.springframework.web.multipart.MultipartFile;

import com.tensor.controllers.BaseRemoteController;
import com.tensor.enums.ErrorType;
import com.tensor.enums.JSONOutputEnum;
import com.tensor.exceptions.VGSException;
import com.tensor.models.Errors;
import com.tensor.models.ImageDetectionTO;
import com.tensor.models.ImageRecognitionTO;
import com.tensor.models.JSONOutputModel;
import com.tensor.services.image.ImageRecognitionService;
import com.tensor.utilities.RestURLConstants;
import com.tensor.utilities.VGSConstants;

@RestController
public class ImageDetectionController extends BaseRemoteController {
	
	@Autowired
	ImageRecognitionService imageRecognitionService;
	
	private static final Logger LOGGER = (Logger) LogManager.getLogger(ImageDetectionController.class);


	/**
	 * To upload data of Terminal entities through file upload
	 * @return JSONOutputModel
	 * @throws VGSException	
	 */
	@RequestMapping(value=RestURLConstants.IMAGE_RECOGNITION,method = RequestMethod.POST)
    @ResponseBody
    public String parseImage(@RequestBody ImageRecognitionTO imageTO) throws VGSException {

		String token=imageRecognitionService.parseImage(imageTO);
		return token;
	}
	
	@RequestMapping(value=RestURLConstants.IMAGE_DATA,method = RequestMethod.GET)
    @ResponseBody
    public JSONOutputModel getImageData(@PathVariable("token") String token) throws VGSException {
		JSONOutputModel output = new JSONOutputModel();
		ImageDetectionTO imageDetectionTO=imageRecognitionService.getImageData(token);
		if(imageDetectionTO!=null){
			output.setData(imageDetectionTO);
			output.setMessage("Image Detected Successfully");
		}
		else{
			output.setData(imageDetectionTO);
			output.setMessage("Image Detection Data not available");
		}
		return output;
	}

}
