import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

/**
 * OpenNLP sentence detection example.
 */
public class SentenceDetectionExample {
  /**
   * Test the sentence detector: parse the document, display found sentences.
   */
  public static void testSentences(String document) {
    String[] sentences = getSentences(document);
    Utils.printStringArray(sentences);
  }

  /**
   * Return the document parsed into sentences, using OpenNLP's sentence detector.
   */
  public static String[] getSentences(String document) {
    SentenceDetectorME detector = new SentenceDetectorME(getSentenceModel());
    String sentences[] = detector.sentDetect(document);
    return sentences;
  }

  /**
   * Return the OpenNLP English sentence model, or null if can't load it.
   */
  private static SentenceModel getSentenceModel() {
    try (InputStream modelIn = new FileInputStream("opennlp-models/en-sent.bin")) {
      return new SentenceModel(modelIn);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }
}
