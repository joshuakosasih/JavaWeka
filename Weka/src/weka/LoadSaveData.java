/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author Jovian
 */
public class LoadSaveData {
    public static void main(String args[]) throws Exception{
        Instances dataset = new Instances(new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff")));
        System.out.println(dataset.toSummaryString());
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("Desktop\\new.arff"));
        saver.writeBatch();
        System.out.println("masuk");
    }
    
}
