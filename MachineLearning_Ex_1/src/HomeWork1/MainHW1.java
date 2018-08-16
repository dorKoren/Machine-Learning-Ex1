package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		// This is the path to the text files.
		String windTraining = "wind_training.txt";
		String windTesting =  "wind_testing.txt";
		
		// Load data from the files.
		Instances trainingData = loadData(windTraining);
		Instances testingData = loadData(windTesting);
		
		LinearRegression lr = new LinearRegression(trainingData);
		
		int numAttributes = trainingData.numAttributes();
		double prevTrainingError = Double.POSITIVE_INFINITY;
		double currTrainingError;
		
		// Initial 3 varaibles to keep the 3 best attributes.
		int attribute1 = 0, attribute2 = 0, attribute3 = 0;
				
		double bestAlpha = lr.getAlpha(trainingData);
		
		
		// Build classifier with all attributes.
		lr.buildClassifier(trainingData);
		
		// Print The chosen best alpha. 
		System.out.println("The chosen alpha is: " + bestAlpha + "\n");
		// Print training error with all features.
		System.out.println("Training error with all features is: " + lr.calculateMSE(trainingData));
		// Print testing error with all features.
		System.out.println("Testing error with all features is: " + lr.calculateMSE(testingData) + "\n");
		
		// Find the best triple attribute.
		for (int i = 0; i < numAttributes - 1; i++) {
			for (int j = i + 1; j < numAttributes - 1; j++) {
				for (int k = j + 1; k < numAttributes - 1; k++) {
					
					trainingData = loadData(windTraining);
					
					// Remove all the unnecessary attributes except for 
					// the current triple attributes.
					for (int l = trainingData.numAttributes() - 2; l >= 0;l--) {
						if ((l != i) && (l != j) && (l != k)) {
							trainingData.deleteAttributeAt(l);
						}
					}
					lr.buildClassifier(trainingData); 
					
					// find the training error of the current triple attributes.
					currTrainingError = lr.calculateMSE(trainingData);
					
					// Print List of all combination of 3 features and the training error.
					System.out.println("attribute 1 = " + 
					trainingData.attribute(0).name() + ",  attribute 2 = " + trainingData.attribute(1).name() + 
					", attribute 3 = " + trainingData.attribute(2).name() + "training error: " + currTrainingError);
					
					// Check if the training error of the current triple 
					// attributes is less than the minimal previous training error.
					if (currTrainingError < prevTrainingError) {
						prevTrainingError = currTrainingError;
						// Update the index of the triple attributes 
						// with respect to the minimal training error
						attribute1 = i; 
						attribute2 = j; 
						attribute3 = k;
					}
				}
			}
		}
		System.out.println();
		
		// Update the training data so that it will contain only the 
		// best triple attributes.
		trainingData = loadData(windTraining);
		for (int i = trainingData.numAttributes() - 2; i >= 0; i--) {
			if ((i != attribute1) && (i != attribute2) && (i != attribute3)) {
				trainingData.deleteAttributeAt(i);
			}
		}
		
		// find best thetas to the best triple attribures.
		lr.buildClassifier(trainingData);
		
		// Print training error the features <best 3 features>. 
		System.out.println("Training error of the best triple features " + 
		trainingData.attribute(0).name() + ", " + trainingData.attribute(1).name() +
		", " + trainingData.attribute(2).name() + "   training error: " + lr.calculateMSE(trainingData));

		// Update the test data so that it will contain only the 
		// best triple attributes.
		for (int i = testingData.numAttributes() - 2; i >= 0; i--) {
			if ((i != attribute1) && (i != attribute2) && (i != attribute3)) {
				testingData.deleteAttributeAt(i);
			}
		}
		
		// Print testing error the features <best 3 features>. 
		System.out.println("Testing error of the best triple features: " +  
		testingData.attribute(0).name() + ", " + testingData.attribute(1).name() +
		", " + testingData.attribute(2).name() + "   testing error: " + lr.calculateMSE(testingData));
	}

}