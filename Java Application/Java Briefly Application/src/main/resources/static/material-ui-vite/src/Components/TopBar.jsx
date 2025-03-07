import { Box, Chip, Container, Divider, Stack } from '@mui/material';
import * as React from 'react';

export default function TopBar({title, children}) {
        return <>
        <Container>
          <Stack
              direction="row"
              spacing={2}
              sx={{
                justifyContent: "space-between",
                alignItems: "center",
                paddingTop:2,
                paddingBottom:1,
                width:"100%"
              }}
            >
              <Box sx={{fontSize: 28 ,
                  fontWeight: 'bold',
              }}>
                {title}
              </Box>
              {children}
          </Stack>
          </Container>
          <Divider></Divider>
        </>
            
  }