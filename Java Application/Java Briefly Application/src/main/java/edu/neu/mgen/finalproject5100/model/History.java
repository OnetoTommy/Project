package edu.neu.mgen.finalproject5100.model;

import lombok.Data;

import java.util.ArrayList;
import java.util.Date;

@Data
public class History {
    @Data
    public static class Item {
        private Article article;
        private Summary summary;
    }

    private ArrayList<Item> items;
    private long timeline;
}
