import { Container, NextUIProvider } from '@nextui-org/react';
import { createTheme } from '@nextui-org/react';

import { WaterForm } from './components/WaterForm';
import { getWaterQuality } from './services/waterQualityService';

export const defaultTheme = createTheme({
  type: 'light',
  theme: {
    colors: {
      gray: '#DDE4E9',
    },
  },
});

export const App = () => {
  return (
    <NextUIProvider theme={defaultTheme}>
      <Container xs>
        <WaterForm onSubmit={getWaterQuality} />
      </Container>
    </NextUIProvider>
  );
};
