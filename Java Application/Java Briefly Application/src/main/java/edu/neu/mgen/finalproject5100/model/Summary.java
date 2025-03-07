package edu.neu.mgen.finalproject5100.model;

import java.util.Date;
// import com.google.cloud.gcp.data.firestore.Document;
import lombok.Data;
import org.springframework.data.annotation.Id;

// @Document(collectionName = "summaries")
@Data
public class Summary {
    @Id
    private String id;
    private String articleId;
    private String originalText;
    private String userSummary;
    private String feedback;
    private int score;
    private String userId; // NEW FIELD
    private Date submissionDate = new Date();

    // Getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    
    public String getArticleId() { return articleId; }
    public void setArticleId(String articleId) { this.articleId = articleId; }
    
    public String getUserSummary() { return userSummary; }
    public void setUserSummary(String userSummary) { this.userSummary = userSummary; }
    
    public Integer getScore() { return score; }
    public void setScore(Integer score) { this.score = score; }
    
    public String getFeedback() { return feedback; }
    public void setFeedback(String feedback) { this.feedback = feedback; }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }


}
