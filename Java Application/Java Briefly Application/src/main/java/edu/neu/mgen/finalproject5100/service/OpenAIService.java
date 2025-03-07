package edu.neu.mgen.finalproject5100.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.service.OpenAiService;

import edu.neu.mgen.finalproject5100.model.Article;
import edu.neu.mgen.finalproject5100.model.Summary;
import edu.neu.mgen.finalproject5100.repository.ArticleRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


@Service
public class OpenAIService {

    @Value("${openai.api.key}")
    private String openaiApiKey;

    // Inject the ArticleRepository to fetch the article
    private final ArticleRepository articleRepository;

    public OpenAIService(ArticleRepository articleRepository) {
        this.articleRepository = articleRepository;
    }

    public Summary evaluateSummary(Summary summary) {
        OpenAiService service = new OpenAiService(openaiApiKey);

        // Fetch the original article content
        Article article = articleRepository.findById(summary.getArticleId());
        if (article == null) {
            summary.setScore(0);
            summary.setFeedback("Original article not found.");
            return summary;
        }

        String articleContent = article.getContent();

        // Update the prompt to include the original article
        String promptContent = String.format("""
            Please evaluate the provided summary based on the original article content and respond as follows:
            1. Provide an overall score (from 0 to 10) based on the summary's ability to accurately and effectively convey the main points of the article. A low-quality or vague summary should receive a correspondingly low score.
            2. Offer constructive feedback with detail on 3 sentences or more based on these criteria with each criteria title in bold:
                Conciseness: Does the summary appropriately condense the article, avoiding unnecessary details while capturing the essence?
                Clarity: Does the summary present specific, coherent, and understandable points? Vague responses should receive lower clarity scores.
                Length: Is the length of the summary adequate to cover the article's main points? Summaries that are overly brief or lack sufficient information should be marked down.
                Readability: Does the summary demonstrate proper grammar, sentence structure, and flow? While readability may be acceptable, lack of content should also impact the overall feedback.
            Article Content:
            %s

            Summary:
            %s

            Provide the evaluation in the following JSON format:
            {
                "score": <number>,
                "feedback": {
                    "conciseness": "<feedback on conciseness>",
                    "clarity": "<feedback on clarity>",
                    "length": "<feedback on length>",
                    "readability": "<feedback on readability>"
                }
            }
            """, articleContent, summary.getUserSummary());

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(new ChatMessage("system", "You are a helpful assistant that evaluates article summaries."));
        messages.add(new ChatMessage("user", promptContent));

        ChatCompletionRequest request = ChatCompletionRequest.builder()
            .model("gpt-3.5-turbo")
            .messages(messages)
            .build();

        String response = service.createChatCompletion(request)
            .getChoices().get(0).getMessage().getContent();

        parseFeedback(response, summary);
        return summary;
    }

    private void parseFeedback(String jsonResponse, Summary summary) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            JsonNode jsonNode = mapper.readTree(jsonResponse);
            summary.setScore(jsonNode.get("score").asInt());
            String feedback = String.format(
                "Conciseness: %s\nClarity: %s\nLength: %s\nReadability: %s",
                jsonNode.get("feedback").get("conciseness").asText(),
                jsonNode.get("feedback").get("clarity").asText(),
                jsonNode.get("feedback").get("length").asText(),
                jsonNode.get("feedback").get("readability").asText()
            );
            summary.setFeedback(feedback);
        } catch (IOException e) {
            e.printStackTrace();
            summary.setScore(0);
            summary.setFeedback("Error parsing feedback from AI response.");
        }
    }
}





// @Service
// public class OpenAIService {
    
//     @Value("${openai.api.key}")
//     private String openaiApiKey;
    
//     public Summary evaluateSummary(Summary summary) {
//         OpenAiService service = new OpenAiService(openaiApiKey);
        
//         // Define the criteria and ask for feedback in a JSON format
//         String promptContent = String.format("""
//             Please evaluate the provided summary of the article and respond as follows:

//             1. Provide an overall score (from 0 to 10) based on the summary's ability to accurately and effectively convey the main points of the article. A low-quality or vague summary should receive a correspondingly low score.
//             2. Offer constructive feedback with detail based on these criteria with each criteria title in bold:
//                 Conciseness: Does the summary appropriately condense the article, avoiding unnecessary details while capturing the essence?
//                 Clarity: Does the summary present specific, coherent, and understandable points? Vague responses should receive lower clarity scores.
//                 Length: Is the length of the summary adequate to cover the article's main points? Summaries that are overly brief or lack sufficient information should be marked down.
//                 Readability: Does the summary demonstrate proper grammar, sentence structure, and flow? While readability may be acceptable, lack of content should also impact the overall feedback.
            
//             Summary:
//             %s

//             Respond in the following JSON format:
//             {
//                 "score": <number>,
//                 "feedback": {
//                     "conciseness": "<feedback on conciseness>",
//                     "clarity": "<feedback on clarity>",
//                     "length": "<feedback on length>",
//                     "readability": "<feedback on readability>"
//                 }
//             }
//             """, summary.getUserSummary());

//         // Create a list of chat messages with the system role
//         List<ChatMessage> messages = new ArrayList<>();
//         messages.add(new ChatMessage("system", "You are a helpful assistant that evaluates article summaries."));
//         messages.add(new ChatMessage("user", promptContent));

//         // Build the chat completion request
//         ChatCompletionRequest request = ChatCompletionRequest.builder()
//             .model("gpt-3.5-turbo")
//             .messages(messages)
//             .build();
            
//         String response = service.createChatCompletion(request)
//             .getChoices().get(0).getMessage().getContent();
            
//         // Parse the JSON response and update the Summary object
//         parseFeedback(response, summary);

//         return summary;
//     }
    
//     private void parseFeedback(String jsonResponse, Summary summary) {
//         ObjectMapper mapper = new ObjectMapper();
//         try {
//             JsonNode jsonNode = mapper.readTree(jsonResponse);
//             // Extract score and feedback details
//             summary.setScore(jsonNode.get("score").asInt());
//             String feedback = String.format(
//                 "Conciseness: %s\nClarity: %s\nLength: %s\nReadability: %s",
//                 jsonNode.get("feedback").get("conciseness").asText(),
//                 jsonNode.get("feedback").get("clarity").asText(),
//                 jsonNode.get("feedback").get("length").asText(),
//                 jsonNode.get("feedback").get("readability").asText()
//             );
//             summary.setFeedback(feedback);
//         } catch (IOException e) {
//             e.printStackTrace();
//             summary.setScore(0);
//             summary.setFeedback("Error parsing feedback from AI response.");
//         }
//     }
// }
