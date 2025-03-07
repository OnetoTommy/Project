package edu.neu.mgen.finalproject5100.model;

import java.util.Date;

import edu.neu.mgen.finalproject5100.util.DateUtil;
import lombok.Data;

// @Document(collectionName = "article")
@Data
public class Article {
    private String id;
    private String title;
    private String content;
    private String poster;
    private Date date;
    private Date createTime = new Date();


    public Article() {
    }

    public Article(String id, Date date, String title, String content, String poster) {
        this.setId(id);
        this.setDate(date);
        this.setTitle(title);
        this.setContent(content);
        this.setPoster(poster);
    }


    public void setDate(Date date) {
        this.date = DateUtil.getStartOfDate(date);
    }


}
