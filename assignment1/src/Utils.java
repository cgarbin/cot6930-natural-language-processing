import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class for OpenNLP test code
 */
public class Utils {
  /**
   * Return the contents of the file to be analyzed as one string, ready to use
   * with the models.
   */
  public static String getSampleText() {
    Path newsArticleFile = Paths.get("testfiles/news article.txt");

    try {
      List<String> lines = Files.readAllLines(newsArticleFile, Charset.defaultCharset());
      return String.join(" ", lines);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }

  /**
   * Print an array with the element index on its left.
   */
  public static void printStringArray(String[] a) {
    int i = 0;
    for (String s : a) {
      System.out.println(String.format("[%02d] %s", i++, s));
    }
  }

  /**
   * Concatenate arrays - from https://stackoverflow.com/a/784842.
   */
  public static <T> T[] concatAll(T[] first, T[]... rest) {
    int totalLength = first.length;
    for (T[] array : rest) {
      totalLength += array.length;
    }
    T[] result = Arrays.copyOf(first, totalLength);
    int offset = first.length;
    for (T[] array : rest) {
      System.arraycopy(array, 0, result, offset, array.length);
      offset += array.length;
    }
    return result;
  }
}
