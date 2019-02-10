import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.Span;

public class NamedEntityExample {

  public static void testNameFinder(String document) {
    // Get the different named entities
    NameFinderME personFinder = new NameFinderME(getNameFinderModel("en-ner-person.bin"));
    NameFinderME locationFinder = new NameFinderME(getNameFinderModel("en-ner-location.bin"));
    NameFinderME moneyFinder = new NameFinderME(getNameFinderModel("en-ner-money.bin"));
    NameFinderME percentageFinder = new NameFinderME(getNameFinderModel("en-ner-percentage.bin"));

    List<TokenizationExample.TokenizedSentence> tokens = TokenizationExample.getMETokens(document);
    for (TokenizationExample.TokenizedSentence ts : tokens) {
      // Get all different types of entities
      Span personSpans[] = personFinder.find(ts.tokens);
      Span locationSpans[] = locationFinder.find(ts.tokens);
      Span moneySpans[] = moneyFinder.find(ts.tokens);
      Span percentageSpans[] = percentageFinder.find(ts.tokens);

      // Combine all entities into one set
      Span[] entities = Utils.concatAll(personSpans, locationSpans, moneySpans, percentageSpans);

      // Remove overlapping spans to simplify this code - in real life we would
      // probably want to know about overlapping spans of different types.
      entities = NameFinderME.dropOverlappingSpans(entities);

      // Show tokens tagged with the named entity, if there is one for that
      // token. This code is not particularly efficient, but it's easy to follow.
      for (int t = 0; t < ts.tokens.length; t++) {
        // Check if this token is the start of a named entity
        for (Span span : entities) {
          if (span.getStart() == t) {
            System.out.print("<" + span.getType().toUpperCase() + "> ");
          }
        }

        // Show the token
        System.out.print(ts.tokens[t] + " ");

        // Check if this token is the end of a named entity
        for (Span span : entities) {
          // End of span is not inclusive, hence the "-1".
          if (span.getEnd() - 1 == t) {
            System.out.print("</" + span.getType().toUpperCase() + "> ");
          }
        }
      }
      // Show next sentence in a new line
      System.out.println("\n");
    }

  }

  /**
   * Return the OpenNLP finder model, given the name of the file.
   */
  private static TokenNameFinderModel getNameFinderModel(String filename) {
    try (InputStream modelIn = new FileInputStream("opennlp-models/" + filename)) {
      return new TokenNameFinderModel(modelIn);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }
}
