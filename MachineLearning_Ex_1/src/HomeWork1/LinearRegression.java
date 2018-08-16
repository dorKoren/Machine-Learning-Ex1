package HomeWork1;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;


public class LinearRegression implements Classifier {
	
	private int m_ClassIndex;
    private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	private static final int MAX_ITERATION =  100; 
	private static final int FIXED_ITERATIONS = 20000; 

	// The method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	public LinearRegression(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1; 
		m_coefficients = new double[m_truNumAttributes + 1];
	}
	
	// The method which runs to train the linear regression predictor.
	// This function assume that the best alpha was calculated.
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1; 
		m_coefficients = new double[m_truNumAttributes + 1];
		m_coefficients = gradientDescent(trainingData);
	}
	
	/**
	 * This function calculate and return the best alpha. 
	 * @param trainingData
	 * @return m_alpha
	 * @throws Exception
	 */
	public double getAlpha(Instances trainingData)throws Exception {
		findAlpha(trainingData);
		return m_alpha;
	}

	/**
	 * Find best alpha for actual training. Test 20 different alphas for 20K
	 * iterations each, and compare resulted errors. Finally, set alpha with
	 * smallest resulted error.
	 * 
	 * @param data
	 * @throws Exception
	 */
	private void findAlpha(Instances data) throws Exception {
		double minAlphaError = 0;
		double minError = Double.MAX_VALUE;
		
		// Iterate over a diffrent values of alphas and 
		// keep the alpha with the lower error.
		for (int i = -17; i <= 0; i++) {
			m_alpha = Math.pow(3, i);
			double currError = calcAlphaError(data);
		
			// If we were unable to compute the error, set it to
			// Double.MAX_VALUE.
			if (Double.isNaN(currError)) {
				currError = Double.MAX_VALUE;
			}

			if (currError < minError) {
				// Update alpha and current error.
				minAlphaError = m_alpha;
				minError = currError;
			}

		}
		this.m_alpha = minAlphaError;
	}
	
	/**
	 * Find the training error with respect to the current alpha
	 * which define at findAlpha function.
	 * @param trainingData
	 * @return training error with respect to the current alpha.
	 * @throws Exception
	 */
	private double calcAlphaError(Instances trainingData) throws Exception {
		
		// Set coefficients to initial values. Assume alpha is predefined.
		Arrays.fill(this.m_coefficients, 1);
		
		double prevError = Double.MAX_VALUE;
		
		// Run gradient descent with respect to the choosen alpha 
		// for 20,000 iterations. 
		for (int i = 0; i < FIXED_ITERATIONS; i++) {
			this.gradientDescentIteration(trainingData);
			
			// Calculate the new error and compare it to the previous error. 
			if (i % 100 == 0) {
				double newError = calculateMSE(trainingData);
				if (newError < prevError) {
					prevError = newError;
				} else break;
			}
		}
		return prevError;
	}

	/**
	 * An implementation of the gradient descent algorithm which should return
	 * the weights of a linear regression predictor which minimizes the average
	 * squared error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		// Set the initial values of the coefficients to be 1
		Arrays.fill(this.m_coefficients, 1);

		// Calculate current error.
		double prevError = Double.MAX_VALUE;
		double currentError = calculateMSE(trainingData);
		double errorRate = Math.abs(currentError - prevError);

		// While the error rate larger than 0.003, update all the training
		// data coefficients by using gradientDecentIteration function.
		while (errorRate > 0.003) {
			for (int i = 0; i < MAX_ITERATION; i++) {
				gradientDescentIteration(trainingData);

			}
			// Update the current test error value
			prevError = currentError;
			currentError = calculateMSE(trainingData);
			errorRate = Math.abs(currentError - prevError);

		}
		return this.m_coefficients;

	}

	/**
	 * An implementation of the gradient descent iteration. This function uses
	 * and updates the this.m_coefficients class member. This function assumes
	 * that the this.m_alpha was predefined.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private void gradientDescentIteration(Instances trainingData) throws Exception {

		double[] newCoefficients = new double[m_coefficients.length];
		// Update each coefficient separately according to the partial
		// derivative with respect to it
		for (int i = 0; i < newCoefficients.length; i++) {
			newCoefficients[i] += this.m_coefficients[i] -  (m_alpha * PartialDerivative(trainingData, i));

		}
		// Update class coefficients
		this.m_coefficients = newCoefficients;

	}
	
	/**
	 * An implementation of the error partial derivative with respect to a
	 * specific coefficient index.
	 * 
	 * @param trainingData
	 * @param coeffIndex
	 * @throws Exception
	 */
	private double PartialDerivative(Instances trainingData, int coeffIndex) throws Exception {
		double sum = 0;

		// Calculate the partial derivative of each Theta.
		for (int i = 0; i < trainingData.numInstances(); i++) {
			Instance currentInstance = trainingData.instance(i);
			double attributeValue;
			if (coeffIndex == trainingData.classIndex()) {
				attributeValue = 1;
			} else {
				attributeValue = currentInstance.value(coeffIndex);

			}
			sum += (regressionPrediction(currentInstance) - currentInstance.classValue()) * attributeValue;

		}
		return (sum / trainingData.numInstances());
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {

		double innerProduct = 0;
		// Calculate the inner product of the coefficients vector and the given instance's attributes vector
		for (int coeffIndex = 0; coeffIndex < this.m_coefficients.length; coeffIndex++) {
			// Set instance's attribute value
			double attributeValue;
			if (coeffIndex == instance.classIndex()) {
				// Use class index as a placeholder for index of Theta0 with attribute value of 1.
				attributeValue = 1;
			} else {
				attributeValue = instance.value(coeffIndex);
			}
			
			innerProduct += m_coefficients[coeffIndex] * attributeValue;
		}
		
		return innerProduct;
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double sum = 0;
		
		for (int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);
			
			sum += Math.pow((regressionPrediction(instance) - instance.classValue()), 2);
		}
		
		return sum / (2 * data.numInstances());
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
