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
import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
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
        
        //filter
        Discretize disc = new Discretize();
        disc.setInputFormat(train);
        Instances filtered = Filter.useFilter(train, disc);
        filtered.setClassIndex(filtered.numAttributes() - 1);
        
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
        
    }
    
}
