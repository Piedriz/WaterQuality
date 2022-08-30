import { Button, Grid, Loading, Row, Spacer, Text } from '@nextui-org/react';
import { styled } from '@stitches/react';
import { FormProvider, useForm } from 'react-hook-form';

import { Field } from './Field';

const CenterContent = styled('div', {
  margin: 'auto',
  display: 'flex',
  minHeight: '100vh',
  flexDirection: 'column',
  justifyContent: 'center',
  background: '$gray',
});

export interface WaterProperties {
  ph: number;
  Hardness: number;
  Solids: number;
  Chloramines: number;
  Sulfate: number;
  Conductivity: number;
  Organic_carbon: number;
  Trihalomethanes: number;
  Turbidity: number;
}

interface WaterFormProps {
  // eslint-disable-next-line no-unused-vars
  onSubmit: (waterProperties: WaterProperties) => Promise<void>;
}

export const WaterForm = ({ onSubmit }: WaterFormProps) => {
  const waterForm = useForm<WaterProperties>({
    mode: 'all',
    reValidateMode: 'onChange',
  });

  return (
    <CenterContent>
      <FormProvider {...waterForm}>
        <form onSubmit={waterForm.handleSubmit(onSubmit)}>
          <Grid.Container justify="center" css={{ paddingInline: '$10' }}>
            <Text h1>Water Quality</Text>
            <Grid xs={12} css={{ marginTop: '$15' }}>
              <Field label="PH" property="ph" />
              <Spacer />
              <Field label="Hardness" property="Hardness" />
            </Grid>
            <Grid xs={12} css={{ marginTop: '$15' }}>
              <Field label="Solids" property="Solids" />
              <Spacer />
              <Field label="Chloramines" property="Chloramines" />
            </Grid>
            <Grid xs={12} css={{ marginTop: '$15' }}>
              <Field label="Sulfate" property="Sulfate" />
              <Spacer />
              <Field label="Conductivity" property="Conductivity" />
            </Grid>
            <Grid xs={12} css={{ marginTop: '$15' }}>
              <Field label="Organic Carbon" property="Organic_carbon" />
              <Spacer />
              <Field label="Trihalomethanes" property="Trihalomethanes" />
            </Grid>
            <Grid xs={12} css={{ marginTop: '$15' }}>
              <Field label="Turbidity" property="Turbidity" />
            </Grid>
          </Grid.Container>

          <Row justify="center" css={{ marginBlock: '$10' }}>
            <Button type="submit" css={{ width: '100%', marginInline: '$10' }} size="lg">
              {waterForm.formState.isSubmitting ? (
                <Loading color="currentColor" type="points-opacity" size="sm" />
              ) : (
                <Text color="currentColor" weight="bold">
                  Verificar
                </Text>
              )}
            </Button>
          </Row>
        </form>
      </FormProvider>
    </CenterContent>
  );
};
