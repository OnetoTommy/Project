package edu.neu.mgen.finalproject5100.controller;

import edu.neu.mgen.finalproject5100.model.Article;
import edu.neu.mgen.finalproject5100.model.ErrorResponse;
import edu.neu.mgen.finalproject5100.model.History;
import edu.neu.mgen.finalproject5100.model.Summary;
import edu.neu.mgen.finalproject5100.repository.ArticleRepository;
import edu.neu.mgen.finalproject5100.repository.SummaryRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

@RestController
@RequestMapping("/api/history")
@CrossOrigin(origins = "*")
public class HistoryController {
    @Autowired
    private SummaryRepository summaryRepository;
    @Autowired
    private ArticleRepository articleRepository;

    @GetMapping("/list")
    public ResponseEntity<?> home(
            @RequestParam(value = "timeline", defaultValue = "0") long timeline,
            @RequestParam("uid") String uid) {

        int limit = 50;
        Date timelineDate;
        timelineDate = timeline == 0 ? new Date() : new Date(timeline);

        List<Summary> summaries = summaryRepository.findAllByUserId(uid, timelineDate, limit);
        if (summaries.isEmpty()) {
            return ResponseEntity.ok(new ArrayList<>());
        }

        ArrayList<String> articleIds = new ArrayList<>();
        summaries.forEach(summary -> {
            articleIds.add(summary.getArticleId());
        });
        ArrayList<Article> articles = articleRepository.findAllByIds(articleIds);

        //convert articles to from ArrayList to HashMap with id as key
        HashMap<String, Article> articleMap = new HashMap<>();
        articles.forEach(article -> {
            articleMap.put(article.getId(), article);
        });


        History history = new History();
        ArrayList<History.Item> historyItems = new ArrayList<>();

        summaries.forEach(summary -> {
            History.Item item = new History.Item();
            item.setSummary(summary);
            item.setArticle(articleMap.get(summary.getArticleId()));
            historyItems.add(item);
        });
        history.setItems(historyItems);

        if (summaries.size() < limit) {
            history.setTimeline(0);
        } else {
            history.setTimeline(summaries.getLast().getSubmissionDate().getTime());
        }

        return ResponseEntity.ok(history);
    }


}
