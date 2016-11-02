/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javaweka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
/**
 *
 * @author Joshua A Kosasih
 */
public class JavaWeka {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        //read file into instance
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("C:/iris.arff"));
        Instances train = new Instances (breader);
        train.setClassIndex(train.numAttributes() - 1);
        
        breader.close();
        
        //System.out.println("Train defiltered: " + train);
        
        //filter
        Discretize disc = new Discretize();
        disc.setInputFormat(train);
        Instances filtered = Filter.useFilter(train, disc);
        filtered.setClassIndex(filtered.numAttributes() - 1);
        
        //System.out.println("Filtered : " + filtered);
        
        //algorithm cross validation
        NaiveBayes nbayes = new NaiveBayes();
        nbayes.buildClassifier(filtered);
        Evaluation eval = new Evaluation(filtered);
        eval.crossValidateModel(nbayes, filtered, 10, new Random(1));
        
        System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
        System.out.println(eval.fMeasure(1)+" "+ eval.precision(1)+" "+ eval.recall(1));
    
        //algorithm train set
        Classifier nb = new NaiveBayes();
        nb.buildClassifier(filtered);
        Evaluation eval2 = new Evaluation(filtered);
        
        eval2.evaluateModel(nb, filtered);
        
        System.out.println(eval2.toSummaryString("\nResults\n=====\n", true));
        System.out.println(eval2.fMeasure(1)+" "+ eval2.precision(1)+" "+ eval2.recall(1));
        
        //create instance from user input
        double[] userdata = new double[4];
        Scanner s = new Scanner(System.in);
        
        System.out.print("Enter sepal length: "); userdata[0] = s.nextFloat();
        System.out.print("Enter sepal width: "); userdata[1] = s.nextFloat();
        System.out.print("Enter petal length: "); userdata[2] = s.nextFloat();
        System.out.print("Enter petal width: "); userdata[3] = s.nextFloat();
        
        Attribute att1 = train.attribute(0);
        Attribute att2 = train.attribute(1);
        Attribute att3 = train.attribute(2);
        Attribute att4 = train.attribute(3);
        //Attribute attC = train.attribute(4);
        
        Instance userins = new DenseInstance(5);
        userins.setDataset(train);
        
        userins.setValue(att1, userdata[0]);
        userins.setValue(att2, userdata[1]);
        userins.setValue(att3, userdata[2]);
        userins.setValue(att4, userdata[3]);
        
        System.out.println("The instance: " + userins);
        
        
        if (disc.input(userins)) {
            Instance ufilter = disc.output();
            System.out.println("The instance: " + ufilter);
            double label = nb.classifyInstance(ufilter);
            System.out.println(train.classAttribute().value((int) label));
        }       
    }   
}
