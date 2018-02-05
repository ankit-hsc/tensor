/**
**********************************************************************************************************
--  FILENAME		: ImageRecognitionService.java
--  DESCRIPTION		: Service Class for Image Recognition
--
--  Copyright		: Copyright (c) 2018.
--  Company			: 
--
--  Revision History
-- --------------------------------------------------------------------------------------------------------
-- |VERSION |      Date                              |      Author              |      Reason for Changes                                         |
-- --------------------------------------------------------------------------------------------------------
-- |  0.1   |   Feb 2, 2018                         |      Ankit Mohanty       |       Initial draft                                                |
-- --------------------------------------------------------------------------------------------------------
--
************************************************************************************************************
**/




package com.tensor.services.image;


import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;

import com.tensor.domains.image.ImageEntity;
import com.tensor.exceptions.VGSException;
import com.tensor.models.ImageDetectionTO;
import com.tensor.models.ImageRecognitionTO;



@Transactional
@Repository
public class ImageRecognitionService {
	
	private static final Logger LOGGER = (Logger) LogManager.getLogger(ImageRecognitionService.class.getName());	
	
	@PersistenceContext	
	private EntityManager entityManager;


	  private static void printUsage(PrintStream s) {
	    final String url =
	        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
	    s.println(
	        "Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)");
	    s.println("to label JPEG images.");
	    s.println("TensorFlow version: " + TensorFlow.version());
	    s.println();
	    s.println("Usage: label_image <model dir> <image file>");
	    s.println();
	    s.println("Where:");
	    s.println("<model dir> is a directory containing the unzipped contents of the inception model");
	    s.println("            (from " + url + ")");
	    s.println("<image file> is the path to a JPEG image file");
	  }

	  // String model, String object, String imageFile
	  public String parseImage(ImageRecognitionTO imageTO) throws VGSException {
	    /*if (args.length != 2) {
	      printUsage(System.err);
	      System.exit(1);
	    }*/
		  
		String url1 = "C:\\Users\\Administrator\\Desktop\\TensorFlow";
		//String url2 = "E:\\TensorFlowProject";
		String result = "";
	    String modelDir = url1;
	    String imageURL = imageTO.getImageURL();
	    
	    byte[] imageBytes = new byte[4096];
	    try {
	    	ByteArrayOutputStream output = new ByteArrayOutputStream();
			URL url = new URL(imageURL);
			InputStream input = url.openStream();
			int length;
			while((length = input.read(imageBytes))!=-1) {
				output.write(imageBytes, 0, length);
			}
			input.close();
			imageBytes = output.toByteArray();
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}
	    catch (IOException e) {
			e.printStackTrace();
			System.out.println("Error while reading the image file");
		}

	    byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, imageTO.getModel()+".pb"));  // Trained model to validate images
	    List<String> labels =  readAllLinesOrExit(Paths.get(modelDir, imageTO.getImageObjectFileName()+".txt")); // sample file contains names of all objects
	//    byte[] imageBytes = readAllBytesOrExit(Paths.get(imageURL,imageFile+ ".jpg")); // image file to upload

	    try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
	      float[] labelProbabilities = executeInceptionGraph(graphDef, image);
	      int bestLabelIdx = maxIndex(labelProbabilities);
	      result = String.format("%s: %.2f%%",
	              labels.get(bestLabelIdx),
	              labelProbabilities[bestLabelIdx] * 100f);
	      System.out.println(
	          String.format("%s: %.2f%%",
	              labels.get(bestLabelIdx),
	              labelProbabilities[bestLabelIdx] * 100f));
	    }
	    result.indexOf(':');
	    ImageEntity imageEntity=new ImageEntity();
	    imageEntity.setObject(result.substring(0,result.indexOf(':')));
	    imageEntity.setProbability(result.substring(result.indexOf(':')+1));
	    imageEntity.setTimeStamp(new Date());	    
	     createImageData(imageEntity);
	     return imageEntity.getToken();
	    
	  }

	  private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
	    try (Graph g = new Graph()) {
	      GraphBuilder b = new GraphBuilder(g);
	      // Some constants specific to the pre-trained model at:
	      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
	      //
	      // - The model was trained with images scaled to 224x224 pixels.
	      // - The colors, represented as R, G, B in 1-byte each were converted to
	      //   float using (value - Mean)/Scale.
	      final int H = 224;
	      final int W = 224;
	      final float mean = 117f;
	      final float scale = 1f;

	      // Since the graph is being constructed once per execution here, we can use a constant for the
	      // input image. If the graph were to be re-used for multiple input images, a placeholder would
	      // have been more appropriate.
	      final Output<String> input = b.constant("input", imageBytes);
	      final Output<Float> output =
	          b.div(
	              b.sub(
	                  b.resizeBilinear(
	                      b.expandDims(
	                          b.cast(b.decodeJpeg(input, 3), Float.class),
	                          b.constant("make_batch", 0)),
	                      b.constant("size", new int[] {H, W})),
	                  b.constant("mean", mean)),
	              b.constant("scale", scale));
	      try (Session s = new Session(g)) {
	        return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
	      }
	    }
	  }

	  private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
	    try (Graph g = new Graph()) {
	      g.importGraphDef(graphDef);
	      try (Session s = new Session(g);
	          Tensor<Float> result =
	              s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class)) {
	        final long[] rshape = result.shape();
	        if (result.numDimensions() != 2 || rshape[0] != 1) {
	          throw new RuntimeException(
	              String.format(
	                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
	                  Arrays.toString(rshape)));
	        }
	        int nlabels = (int) rshape[1];
	        return result.copyTo(new float[1][nlabels])[0];
	      }
	    }
	  }

	  private static int maxIndex(float[] probabilities) {
	    int best = 0;
	    for (int i = 1; i < probabilities.length; ++i) {
	      if (probabilities[i] > probabilities[best]) {
	        best = i;
	      }
	    }
	    return best;
	  }

	  private static byte[] readAllBytesOrExit(Path path) {
	    try {
	      return Files.readAllBytes(path);
	    } catch (IOException e) {
	      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
	      System.exit(1);
	    }
	    return null;
	  }

	  private static List<String> readAllLinesOrExit(Path path) {
	    try {
	      return Files.readAllLines(path, Charset.forName("UTF-8"));
	    } catch (IOException e) {
	      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
	      System.exit(0);
	    }
	    return null;
	  }

	  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
	  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
	  // like Python, C++ and Go.
	  static class GraphBuilder {
	    GraphBuilder(Graph g) {
	      this.g = g;
	    }

	    Output<Float> div(Output<Float> x, Output<Float> y) {
	      return binaryOp("Div", x, y);
	    }

	    <T> Output<T> sub(Output<T> x, Output<T> y) {
	      return binaryOp("Sub", x, y);
	    }

	    <T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
	      return binaryOp3("ResizeBilinear", images, size);
	    }

	    <T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
	      return binaryOp3("ExpandDims", input, dim);
	    }

	    <T, U> Output<U> cast(Output<T> value, Class<U> type) {
	      DataType dtype = DataType.fromClass(type);
	      return g.opBuilder("Cast", "Cast")
	          .addInput(value)
	          .setAttr("DstT", dtype)
	          .build()
	          .<U>output(0);
	    }

	    Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
	      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
	          .addInput(contents)
	          .setAttr("channels", channels)
	          .build()
	          .<UInt8>output(0);
	    }

	    <T> Output<T> constant(String name, Object value, Class<T> type) {
	      try (Tensor<T> t = Tensor.<T>create(value, type)) {
	        return g.opBuilder("Const", name)
	            .setAttr("dtype", DataType.fromClass(type))
	            .setAttr("value", t)
	            .build()
	            .<T>output(0);
	      }
	    }
	    Output<String> constant(String name, byte[] value) {
	      return this.constant(name, value, String.class);
	    }

	    Output<Integer> constant(String name, int value) {
	      return this.constant(name, value, Integer.class);
	    }

	    Output<Integer> constant(String name, int[] value) {
	      return this.constant(name, value, Integer.class);
	    }

	    Output<Float> constant(String name, float value) {
	      return this.constant(name, value, Float.class);
	    }

	    private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
	      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
	    }

	    private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
	      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
	    }
	    private Graph g;
	  }

	
	/**
	 * To create image recognition data
	 * @return void
	 * @throws VGSException
	 * 
	 */
	
	public boolean createImageData(ImageEntity entity) throws VGSException{
		try{
			entityManager.persist(entity);
		}
		catch (Exception e) {
			throw new VGSException("Error creating image data", e);
		}
		return true;
	}
	
	public ImageDetectionTO getImageData(String token) throws VGSException{
		ImageDetectionTO imageDetectionTO=new ImageDetectionTO();
		try{
			List<ImageEntity> imageEntityList =entityManager.createQuery("select image from ImageEntity image where image.token=:id").setParameter("id", token).getResultList();
			if(imageEntityList!=null && !imageEntityList.isEmpty()){
				imageDetectionTO.setObject(imageEntityList.get(0).getObject());
				imageDetectionTO.setProbability(imageEntityList.get(0).getProbability()+"");
				imageDetectionTO.setToken(imageEntityList.get(0).getToken());
			}
		}
		catch (Exception e) {
			throw new VGSException("Error fetching image data", e);
		}
		return imageDetectionTO;
	}

	}
